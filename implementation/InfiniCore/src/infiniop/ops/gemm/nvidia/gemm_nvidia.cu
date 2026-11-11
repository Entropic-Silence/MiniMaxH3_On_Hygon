#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "gemm_nvidia.cuh"
// cuBLASLt BF16 path: NVIDIA always; Hygon on request (DTK maps the cuBLASLt
// API onto hipBLASLt), INFINICORE_HYGON_GEMM=lt.
#if !defined(ENABLE_ILUVATAR_API)
#define INFINIOP_GEMM_LT 1
#include <cublasLt.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#endif

namespace op::gemm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
#if defined(INFINIOP_GEMM_LT)
    cublasLtHandle_t lt_handle = nullptr;
    cublasLtMatmulDesc_t lt_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    // Explicit heuristic choice: DTK's cuBLASLt (hipBLASLt) needs an algo.
    cublasLtMatmulAlgo_t lt_algo{};
    bool lt_has_algo = false;
    size_t lt_workspace = 0;
    // Offline tuning (INFINICORE_GEMM_LT_AUTOTUNE=1): every heuristic
    // candidate, timed on the first call; see Opaque::autotune.
    std::string lt_key;
    std::vector<cublasLtMatmulHeuristicResult_t> lt_candidates;
    bool lt_tuned = true;
    void autotune(const MatmulInfo &info, const void *a, const void *b, const void *c,
                  const float *alpha, const float *beta, void *workspace, size_t workspace_size,
                  cudaStream_t stream);

    void destroyLtDescriptors();
    bool createBf16LtDescriptors(const MatmulInfo &info);
#endif
};

#if defined(INFINIOP_GEMM_LT)
static bool ltEnabled() {
#if defined(ENABLE_HYGON_API)
    const char *mode = std::getenv("INFINICORE_HYGON_GEMM");
    return mode != nullptr && std::strcmp(mode, "lt") == 0;
#else
    return true;
#endif
}

static size_t ltLayoutRows(const BlasMatrix &matrix) {
    return matrix.row_stride == 1 ? matrix.rows : matrix.cols;
}

static size_t ltLayoutCols(const BlasMatrix &matrix) {
    return matrix.row_stride == 1 ? matrix.cols : matrix.rows;
}

static bool setLtLayoutBatch(cublasLtMatrixLayout_t layout, const BlasMatrix &matrix, size_t batch) {
    int32_t batch_count = static_cast<int32_t>(batch);
    int64_t stride = static_cast<int64_t>(matrix.stride);
    return cublasLtMatrixLayoutSetAttribute(
               layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
               &batch_count, sizeof(batch_count))
            == CUBLAS_STATUS_SUCCESS
        && cublasLtMatrixLayoutSetAttribute(
               layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
               &stride, sizeof(stride))
               == CUBLAS_STATUS_SUCCESS;
}

void Descriptor::Opaque::destroyLtDescriptors() {
    if (a_layout) {
        cublasLtMatrixLayoutDestroy(a_layout);
        a_layout = nullptr;
    }
    if (b_layout) {
        cublasLtMatrixLayoutDestroy(b_layout);
        b_layout = nullptr;
    }
    if (c_layout) {
        cublasLtMatrixLayoutDestroy(c_layout);
        c_layout = nullptr;
    }
    if (lt_desc) {
        cublasLtMatmulDescDestroy(lt_desc);
        lt_desc = nullptr;
    }
    if (lt_handle) {
        cublasLtDestroy(lt_handle);
        lt_handle = nullptr;
    }
}

// Per-shape choice among hipBLASLt's heuristic candidates.
//
// cublasLtMatmulAlgoGetHeuristic returns candidates in the library's
// predicted order; the first is not always the fastest on a power-capped
// card. INFINICORE_GEMM_LT_TABLE names a text file of offline-measured
// choices, one per line: "<device> <ops> m n k batch lda ldb ldc = <index>"
// ('#' starts a comment). Shapes not listed use candidate 0. The table is
// produced by running the workload once with INFINICORE_GEMM_LT_AUTOTUNE=1,
// which times every candidate on the first call of each shape and prints the
// table line to stderr (tools/tune_gemm_lt.py collects them).
static constexpr int kLtMaxCandidates = 16;

static bool ltAutotuneEnabled() {
    const char *v = std::getenv("INFINICORE_GEMM_LT_AUTOTUNE");
    return v != nullptr && std::strcmp(v, "1") == 0;
}

static std::string ltDeviceName() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return "unknown";
    }
    std::string name(prop.name);
    for (auto &ch : name) {
        if (ch == ' ') {
            ch = '_';
        }
    }
    return name.empty() ? "unknown" : name;
}

static std::string ltKey(const MatmulInfo &info) {
    std::ostringstream key;
    key << ltDeviceName() << ' '
        << (info.a_matrix.row_stride == 1 ? 'N' : 'T') << (info.b_matrix.row_stride == 1 ? 'N' : 'T')
        << ' ' << info.m << ' ' << info.n << ' ' << info.k << ' ' << info.batch << ' '
        << info.a_matrix.ld() << ' ' << info.b_matrix.ld() << ' ' << info.c_matrix.ld();
    return key.str();
}

static const std::map<std::string, int> &ltTable() {
    static std::map<std::string, int> table;
    static std::once_flag once;
    std::call_once(once, [] {
        const char *path = std::getenv("INFINICORE_GEMM_LT_TABLE");
        if (path == nullptr || *path == '\0') {
            return;
        }
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            line = line.substr(0, line.find('#'));
            const auto eq = line.find('=');
            if (eq == std::string::npos) {
                continue;
            }
            std::istringstream fields(line.substr(0, eq));
            std::string field, key;
            while (fields >> field) {
                key += (key.empty() ? "" : " ") + field;
            }
            table[key] = std::atoi(line.c_str() + eq + 1);
        }
    });
    return table;
}

bool Descriptor::Opaque::createBf16LtDescriptors(const MatmulInfo &info) {
    auto op_a = info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    if (cublasLtCreate(&lt_handle) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescCreate(&lt_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(
            lt_desc, CUBLASLT_MATMUL_DESC_TRANSA,
            &op_a, sizeof(op_a))
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(
            lt_desc, CUBLASLT_MATMUL_DESC_TRANSB,
            &op_b, sizeof(op_b))
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    if (cublasLtMatrixLayoutCreate(
            &a_layout, CUDA_R_16BF,
            ltLayoutRows(info.a_matrix), ltLayoutCols(info.a_matrix),
            info.a_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatrixLayoutCreate(
            &b_layout, CUDA_R_16BF,
            ltLayoutRows(info.b_matrix), ltLayoutCols(info.b_matrix),
            info.b_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatrixLayoutCreate(
            &c_layout, CUDA_R_16BF,
            ltLayoutRows(info.c_matrix), ltLayoutCols(info.c_matrix),
            info.c_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    if (!(setLtLayoutBatch(a_layout, info.a_matrix, info.batch)
          && setLtLayoutBatch(b_layout, info.b_matrix, info.batch)
          && setLtLayoutBatch(c_layout, info.c_matrix, info.batch))) {
        return false;
    }

    // Best heuristic algorithm within a workspace limit (MiB), chosen once
    // per descriptor (descriptors are cached per shape).
    size_t workspace_limit = size_t(32) << 20;
    if (const char *mb = std::getenv("INFINICORE_GEMM_LT_WORKSPACE_MB")) {
        workspace_limit = static_cast<size_t>(std::strtoull(mb, nullptr, 10)) << 20;
    }
    cublasLtMatmulPreference_t preference = nullptr;
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    bool ok = cublasLtMatmulPreferenceSetAttribute(
                  preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &workspace_limit, sizeof(workspace_limit))
           == CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulHeuristicResult_t results[kLtMaxCandidates]{};
    int found = 0;
    ok = ok
      && cublasLtMatmulAlgoGetHeuristic(lt_handle, lt_desc, a_layout, b_layout, c_layout, c_layout,
                                        preference, kLtMaxCandidates, results, &found)
             == CUBLAS_STATUS_SUCCESS
      && found > 0;
    cublasLtMatmulPreferenceDestroy(preference);
    if (!ok) {
        return false;
    }
    lt_key = ltKey(info);
    int choice = 0;
    const auto &table = ltTable();
    if (auto it = table.find(lt_key); it != table.end() && it->second >= 0 && it->second < found) {
        choice = it->second;
    }
    lt_algo = results[choice].algo;
    lt_workspace = results[choice].workspaceSize;
    lt_has_algo = true;
    if (ltAutotuneEnabled()) {
        lt_candidates.assign(results, results + found);
        for (const auto &candidate : lt_candidates) {
            lt_workspace = std::max(lt_workspace, candidate.workspaceSize);
        }
        lt_tuned = false;
    }
    return true;
}

// Times every candidate on the caller's operands, writing into scratch
// outputs (C is only read), and keeps the fastest one whose result is
// bitwise reproducible between two runs (no atomics-based split-K). Prints
// "<key> = <index>  # index:ms ..." to stderr. Diagnostic/offline only.
void Descriptor::Opaque::autotune(const MatmulInfo &info, const void *a, const void *b, const void *c,
                                  const float *alpha, const float *beta, void *workspace,
                                  size_t workspace_size, cudaStream_t stream) {
    Opaque *o = this;
    o->lt_tuned = true;
    const auto &cm = info.c_matrix;
    const size_t cols = cm.row_stride == 1 ? cm.cols : cm.rows;
    const size_t elems = static_cast<size_t>(cm.ld()) * cols
                       + (info.batch > 1 ? static_cast<size_t>(cm.stride) * (info.batch - 1) : 0);
    const size_t bytes = elems * 2;
    void *d0 = nullptr, *d1 = nullptr;
    if (cudaMalloc(&d0, bytes) != cudaSuccess || cudaMalloc(&d1, bytes) != cudaSuccess) {
        cudaFree(d0);
        std::fprintf(stderr, "[infinicore.gemm_lt] %s autotune skipped: no scratch memory\n", o->lt_key.c_str());
        return;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<char> h0(bytes), h1(bytes);
    auto run = [&](const cublasLtMatmulHeuristicResult_t &cand, void *d) {
        return cublasLtMatmul(o->lt_handle, o->lt_desc, alpha, a, o->a_layout, b, o->b_layout, beta,
                              c, o->c_layout, d, o->c_layout, &cand.algo, workspace, workspace_size, stream);
    };
    int best = -1;
    float best_ms = 0.0f;
    std::ostringstream report;
    constexpr int repeats = 5;
    for (size_t i = 0; i < o->lt_candidates.size(); ++i) {
        const auto &cand = o->lt_candidates[i];
        report << ' ' << i << ':';
        if (cand.workspaceSize > workspace_size || run(cand, d0) != CUBLAS_STATUS_SUCCESS) {
            report << "fail";
            continue;
        }
        cudaEventRecord(start, stream);
        for (int r = 0; r < repeats; ++r) {
            run(cand, d0);
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= repeats;
        run(cand, d1);
        cudaStreamSynchronize(stream);
        cudaMemcpy(h0.data(), d0, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h1.data(), d1, bytes, cudaMemcpyDeviceToHost);
        const bool reproducible = std::memcmp(h0.data(), h1.data(), bytes) == 0;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f%s", ms, reproducible ? "" : "(nondet)");
        report << buf;
        if (reproducible && (best < 0 || ms < best_ms)) {
            best = static_cast<int>(i);
            best_ms = ms;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d0);
    cudaFree(d1);
    if (best >= 0) {
        o->lt_algo = o->lt_candidates[best].algo;
    }
    std::fprintf(stderr, "[infinicore.gemm_lt] %s = %d  #%s\n", o->lt_key.c_str(), best < 0 ? 0 : best,
                 report.str().c_str());
}
#endif

Descriptor::~Descriptor() {
#if defined(INFINIOP_GEMM_LT)
    if (_opaque) {
        _opaque->destroyLtDescriptors();
    }
#endif
    delete _opaque;
}

#if defined(INFINIOP_GEMM_LT)
// Opt-in experiment. Only layouts/shapes screened on BW gfx936 are eligible.
// Keep Lt for gate/up projections and every unmeasured case. The caller must
// set the option before descriptor creation; cached descriptors do not change.
static bool h3SelectiveBlas(const MatmulInfo &info) {
#if defined(ENABLE_HYGON_API)
    const char *mode = std::getenv("H3_BW_GEMM_SELECTIVE");
    if (!mode || std::strcmp(mode, "1") != 0) return false;
    if (ltDeviceName().find("BW_gfx936") != 0) return false;
    // Opt-in coverage beyond the two original prompt lengths. Single-card
    // screening passed 640 comparisons across 80 lengths in this range.
    // Keep the original whitelist when disabled, and all layout/shape guards.
    const char *lengths = std::getenv("H3_BW_GEMM_LENGTHS");
    const bool broad_lengths = lengths && std::strcmp(lengths, "1") == 0;
    const bool eligible_length = broad_lengths ? (info.n >= 512 && info.n <= 8192)
                                              : (info.n == 2862 || info.n == 588);
    // Direct Lt/BLAS NN comparisons passed for both gate token counts.
    // This extra switch is independent of the previously validated TN cases.
    const char *gate_blas = std::getenv("H3_BW_GATE_BLAS");
    if (gate_blas && std::strcmp(gate_blas, "1") == 0
        && info.batch == 1 && eligible_length
        && info.m == 28672 && info.k == 5376
        && info.a_matrix.row_stride == 1 && info.b_matrix.row_stride == 1
        && info.c_matrix.row_stride == 1 && info.a_matrix.ld() == info.m
        && info.b_matrix.ld() == info.k && info.c_matrix.ld() == info.m) return true;
    if (info.batch != 1 || !eligible_length
        || info.a_matrix.row_stride == 1 || info.b_matrix.row_stride != 1
        || info.c_matrix.row_stride != 1 || info.a_matrix.ld() != info.k
        || info.b_matrix.ld() != info.k || info.c_matrix.ld() != info.m) return false;
    return (info.m == 21504 && info.k == 5376)
        || (info.m == 5376 && (info.k == 14336 || info.k == 7168));
#else
    return false;
#endif
}
#endif

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    auto info = result.take();
    auto opaque = new Opaque{handle->internal()};
    size_t workspace_size = 0;
#if defined(INFINIOP_GEMM_LT)
    if (dtype == INFINI_DTYPE_BF16 && ltEnabled() && !h3SelectiveBlas(info)) {
        if (opaque->createBf16LtDescriptors(info)) {
            workspace_size = opaque->lt_workspace;
        } else {
            opaque->destroyLtDescriptors();
            opaque->lt_has_algo = false;
        }
    }
#endif

    *desc_ptr = new Descriptor(
        dtype, info, workspace_size,
        opaque,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    cudaDataType a_type, b_type, c_type;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
    cudaDataType compute_type;
#else
    cublasComputeType_t compute_type;
#endif

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        a_type = b_type = c_type = CUDA_R_16F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_BF16:
        a_type = b_type = c_type = CUDA_R_16BF;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_F32:
        a_type = b_type = c_type = CUDA_R_32F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#endif
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (_info.is_transed) {
        std::swap(a, b);
    }

    auto op_a = _info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = _info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

#if defined(INFINIOP_GEMM_LT)
    if (_dtype == INFINI_DTYPE_BF16 && _opaque->lt_handle && _opaque->lt_desc
        && _opaque->a_layout && _opaque->b_layout && _opaque->c_layout && _opaque->lt_has_algo
        && workspace_size >= _opaque->lt_workspace) {
        if (!_opaque->lt_tuned) {
            _opaque->autotune(_info, a, b, c, &alpha, &beta, workspace, workspace_size,
                              (cudaStream_t)stream);
        }
        auto lt_status = cublasLtMatmul(
            _opaque->lt_handle,
            _opaque->lt_desc,
            &alpha,
            a,
            _opaque->a_layout,
            b,
            _opaque->b_layout,
            &beta,
            c,
            _opaque->c_layout,
            c,
            _opaque->c_layout,
            &_opaque->lt_algo,
            workspace,
            workspace_size,
            (cudaStream_t)stream);
        if (lt_status == CUBLAS_STATUS_SUCCESS) {
            return INFINI_STATUS_SUCCESS;
        }
    }
#endif

    CHECK_STATUS(_opaque->internal->useCublas(
        (cudaStream_t)stream,
        [&](cublasHandle_t handle) {
            CHECK_CUBLAS(
                cublasGemmStridedBatchedEx(
                    handle,
                    op_a,
                    op_b,
                    static_cast<int>(_info.m),
                    static_cast<int>(_info.n),
                    static_cast<int>(_info.k),
                    &alpha,
                    a,
                    a_type,
                    static_cast<int>(_info.a_matrix.ld()),
                    _info.a_matrix.stride,
                    b,
                    b_type,
                    static_cast<int>(_info.b_matrix.ld()),
                    _info.b_matrix.stride,
                    &beta,
                    c,
                    c_type,
                    static_cast<int>(_info.c_matrix.ld()),
                    _info.c_matrix.stride,
                    static_cast<int>(_info.batch),
                    compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::nvidia
