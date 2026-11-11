// Isolated native C ABI backend. Opt-in only; accepted ATen fallback is retained.
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <hip/hip_runtime_api.h>

namespace {
using H3DenseForward = int (*)(void *, void *, void *, void *, void *,
    int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    float, void *);
std::atomic<unsigned long long> h3_vendor_counts[64][5]{};
std::atomic<int> h3_dense_arch[64]{};

struct H3DenseHandle {
    void *handle = nullptr;
    H3DenseForward forward = nullptr;
    std::string path;
    H3DenseHandle() {
        const char *requested = std::getenv("H3_DENSE_TILE_LIBRARY");
        if (!requested || requested[0] != '/')
            throw std::invalid_argument("H3_DENSE_TILE_LIBRARY requires an absolute path");
        char *resolved = realpath(requested, nullptr);
        if (!resolved) throw std::runtime_error("Cannot resolve dense kernel library");
        path = resolved; std::free(resolved);
        if (path != requested) throw std::invalid_argument("Dense library path must be canonical");
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (!handle) throw std::runtime_error("Cannot open dense kernel library");
        auto get = [&](const char *name) {
            void *p = dlsym(handle, name); Dl_info info{};
            if (!p || !dladdr(p, &info) || !info.dli_fname)
                throw std::runtime_error("Missing dense kernel symbol");
            char *actual = realpath(info.dli_fname, nullptr);
            bool match = actual && path == actual; std::free(actual);
            if (!match) throw std::runtime_error("Dense symbol resolved to unexpected library");
            return p;
        };
        auto block_m = reinterpret_cast<int (*)()>(get("h3_dense_tile_block_m"));
        auto stages = reinterpret_cast<int (*)()>(get("h3_dense_tile_stages"));
        auto round = reinterpret_cast<int (*)()>(get("h3_dense_tile_round"));
        if ((block_m() != 64 && block_m() != 128 && block_m() != 192)
            || stages() != 3 || round() != 1)
            throw std::runtime_error("Unexpected dense kernel metadata");
        forward = reinterpret_cast<H3DenseForward>(get("h3_dense_tile_forward"));
    }
    // Keep code resident until process exit, including outstanding GPU work.
};

bool h3_try_vendor_route(at::Tensor &q, const at::Tensor &k,
                         const at::Tensor &v, at::Tensor &out,
                         float scale, bool causal, bool has_alibi) {
    const char *enabled = std::getenv("H3_DENSE_TILE_ENABLE");
    if (enabled && std::strcmp(enabled,"0") != 0 && std::strcmp(enabled,"1") != 0)
        throw std::invalid_argument("H3_DENSE_TILE_ENABLE must be 0 or 1");
    const int device = q.get_device();
    if (device < 0 || device >= 64) throw std::runtime_error("Invalid dense route device");
    const bool eligible = !causal && !has_alibi
        && q.dim()==4 && k.dim()==4 && v.dim()==4
        && q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16
        && v.scalar_type()==at::kBFloat16 && out.scalar_type()==at::kBFloat16
        && q.size(0)==1 && k.size(0)==1 && v.size(0)==1
        && q.size(2)==7 && k.size(2)==7 && v.size(2)==7
        && q.size(3)==128 && k.size(3)==128 && v.size(3)==128
        && k.sizes()==v.sizes() && q.sizes()==out.sizes()
        && q.stride(3)==1 && k.stride(3)==1 && v.stride(3)==1
        && q.stride(2)==128 && k.stride(2)==128 && v.stride(2)==128
        && q.stride(1)%8==0 && k.stride(1)%8==0 && v.stride(1)%8==0
        && q.device()==k.device() && q.device()==v.device() && q.device()==out.device()
        && out.is_contiguous();
    const bool long_sequence = eligible && q.size(1)>=4096 && k.size(1)>=4096
        && q.size(1)<=2147483647LL && k.size(1)<=2147483647LL;
    bool use_new = enabled && std::strcmp(enabled,"1")==0 && long_sequence;
    if (use_new) {
        int architecture = h3_dense_arch[device].load(std::memory_order_relaxed);
        if (!architecture) {
            hipDeviceProp_t properties{};
            if (hipGetDeviceProperties(&properties,device)!=hipSuccess)
                throw std::runtime_error("Cannot inspect dense kernel device");
            architecture = std::strncmp(properties.gcnArchName,"gfx936",6)==0 ? 1 : -1;
            h3_dense_arch[device].store(architecture,std::memory_order_relaxed);
        }
        use_new = architecture == 1;
    }
    if (!use_new) {
        h3_vendor_counts[device][eligible ? (long_sequence ? 0 : 1) : 4].fetch_add(1,std::memory_order_relaxed);
        return false;
    }
    const char *fast = std::getenv("FLASH_ATTENTION_FAST_SOFTMAX");
    if (!fast || std::strcmp(fast,"0")!=0)
        throw std::invalid_argument("Dense route requires ordinary softmax");
    static H3DenseHandle kernel;
    const char *requested = std::getenv("H3_DENSE_TILE_LIBRARY");
    if (!requested || kernel.path!=requested)
        throw std::invalid_argument("Dense library must remain fixed within a process");
    auto lse = at::empty({1,7,q.size(1)},q.options().dtype(at::kFloat));
    auto stream = infinicore::adaptor::get_hip_stream();
    const int code = kernel.forward(q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),lse.data_ptr(),
        int(q.size(1)),int(k.size(1)),7,q.stride(1),q.stride(2),k.stride(1),k.stride(2),
        v.stride(1),v.stride(2),scale,reinterpret_cast<void *>(stream.stream()));
    if (code != 0) throw std::runtime_error("Dense attention kernel launch failed");
    h3_vendor_counts[device][2].fetch_add(1,std::memory_order_relaxed);
    return true;
}
} // namespace
extern "C" unsigned long long h3_vendor_fa_route_counter(int device,int bucket) {
    if(device<0 || device>=64 || bucket<0 || bucket>=5) return ~0ULL;
    return h3_vendor_counts[device][bucket].load(std::memory_order_relaxed);
}
extern "C" void h3_vendor_fa_route_reset() {
    for(auto &device:h3_vendor_counts) for(auto &count:device) count.store(0,std::memory_order_relaxed);
}
