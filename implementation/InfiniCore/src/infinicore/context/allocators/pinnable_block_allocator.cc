#include "pinnable_block_allocator.hpp"

#include "../context_impl.hpp"

#include "../../utils.hpp"

#include <algorithm>
#include <infinirt.h>
#include <stdexcept>

namespace infinicore {

// ------------------- Helper functions -------------------

// Round up size to nearest multiple of alignment
inline size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

namespace {
// Requests above the largest size class are served by exact-size large blocks
// rounded to this granularity. Power-of-two classes above 32 MB wasted up to
// half of each block (e.g. 130 MB weight shards occupying 256 MB blocks).
constexpr size_t LARGE_BLOCK_GRANULARITY = 2 * 1024 * 1024;

// A free large block is reused only when it is not much larger than the
// request, so a big cached block is not tied up by a small tensor.
bool acceptable_reuse(size_t block_size, size_t request) {
    return block_size >= request && block_size - request <= std::max(request / 8, LARGE_BLOCK_GRANULARITY);
}
} // namespace

// ------------------- Constructor -------------------
PinnableBlockAllocator::PinnableBlockAllocator(Device device)
    : device_(device) {
    size_classes_ = {
        {32 * 1024, {}},        // 32 KB
        {256 * 1024, {}},       // 256 KB
        {1 * 1024 * 1024, {}},  // 1 MB
        {2 * 1024 * 1024, {}},  // 2 MB
        {4 * 1024 * 1024, {}},  // 4 MB
        {8 * 1024 * 1024, {}},  // 8 MB
        {16 * 1024 * 1024, {}}, // 16 MB
        {32 * 1024 * 1024, {}}, // 32 MB
    };
}

// ------------------- allocate -------------------
std::byte *PinnableBlockAllocator::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);

    // Align size to 256 bytes for GPU
    size = align_up(size, 256);

    std::shared_ptr<Block> block;

    // 1. Try size-class allocation for small/medium
    for (auto &cls : size_classes_) {
        if (size <= cls.block_size) {
            if (!cls.free_blocks.empty()) {
                block = cls.free_blocks.back();
                while (block != nullptr && block->in_use) {
                    cls.free_blocks.pop_back();
                    if (cls.free_blocks.empty()) {
                        block = nullptr;
                        break;
                    }
                    block = cls.free_blocks.back();
                }
                if (block != nullptr) {
                    cls.free_blocks.pop_back();
                    block->in_use = true;
                    block->use_count = 1;
                    return reinterpret_cast<std::byte *>(block->ptr);
                }
            }
            // Allocate a new block for this class
            block = std::make_shared<Block>();
            block->size = cls.block_size;
            block->frozen = pinned_mode_;
            block->in_use = true;
            block->use_count = 1;

            device_malloc_(&block->ptr, block->size);

            all_blocks_[block->ptr] = block;
            return reinterpret_cast<std::byte *>(block->ptr);
        }
    }

    // 2. Large block allocation
    size = align_up(size, LARGE_BLOCK_GRANULARITY);
    // Reuse the best-fitting free large block, if it is close enough in size
    auto it = large_blocks_.end();
    for (auto candidate = large_blocks_.begin(); candidate != large_blocks_.end(); ++candidate) {
        if (!(*candidate)->in_use && acceptable_reuse((*candidate)->size, size)
            && (it == large_blocks_.end() || (*candidate)->size < (*it)->size)) {
            it = candidate;
        }
    }

    if (it != large_blocks_.end()) {
        block = *it;
        block->in_use = true;
        block->use_count = 1;
        block->frozen = block->frozen || pinned_mode_;
        return reinterpret_cast<std::byte *>(block->ptr);
    }

    // Allocate new large block
    block = std::make_shared<Block>();
    block->size = size;
    block->large = true;
    block->frozen = pinned_mode_;
    block->in_use = true;
    block->use_count = 1;

    device_malloc_(&block->ptr, block->size);

    large_blocks_.push_back(block);
    all_blocks_[block->ptr] = block;

    return reinterpret_cast<std::byte *>(block->ptr);
}

// ------------------- deallocate -------------------
void PinnableBlockAllocator::deallocate(std::byte *ptr) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = all_blocks_.find(reinterpret_cast<void *>(ptr));
    if (it == all_blocks_.end()) {
        throw std::runtime_error("Pointer not allocated by this allocator");
    }

    auto block = it->second;
    if (!block->in_use || block->use_count == 0) {
        throw std::runtime_error("Double free detected in PinnableBlockAllocator");
    }

    --block->use_count;
    if (block->use_count > 0) {
        return;
    }

    block->in_use = false;
    if (block->large) {
        // Large blocks are found through large_blocks_; their sizes may equal
        // a class size, so they must never enter a class free list.
        return;
    }
    for (auto &cls : size_classes_) {
        if (block->size == cls.block_size) {
            cls.free_blocks.push_back(block);
            break;
        }
    }
}

size_t PinnableBlockAllocator::mark_in_use_(void *ptr, bool in_use) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = all_blocks_.find(reinterpret_cast<void *>(ptr));
    if (it == all_blocks_.end()) {
        throw std::runtime_error("Pointer not allocated by this allocator");
    }

    auto block = it->second;
    if (in_use) {
        block->in_use = true;
        ++block->use_count;
    } else if (block->use_count > 0) {
        --block->use_count;
        block->in_use = block->use_count > 0;
    }
    return it->second->size;
}

// Device allocation; when the device is out of memory, cached blocks that are
// not in use are returned first and the allocation is retried once.
void PinnableBlockAllocator::device_malloc_(void **ptr, size_t size) {
    if (infinirtMalloc(ptr, size) == INFINI_STATUS_SUCCESS) {
        return;
    }
    trim_locked_();
    INFINICORE_CHECK_ERROR(infinirtMalloc(ptr, size));
}

// ------------------- trim -------------------
void PinnableBlockAllocator::trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    trim_locked_();
}

void PinnableBlockAllocator::trim_locked_() {
    // Free non-frozen size-class blocks (entries re-marked in use stay)
    for (auto &cls : size_classes_) {
        for (auto it = cls.free_blocks.begin(); it != cls.free_blocks.end();) {
            if (!(*it)->frozen && !(*it)->in_use) {
                INFINICORE_CHECK_ERROR(infinirtFree((*it)->ptr));
                all_blocks_.erase((*it)->ptr);
                it = cls.free_blocks.erase(it);
            } else {
                ++it;
            }
        }
    }
    // Free non-frozen large blocks
    for (auto it = large_blocks_.begin(); it != large_blocks_.end();) {
        if (!(*it)->frozen && !(*it)->in_use) {
            INFINICORE_CHECK_ERROR(infinirtFree((*it)->ptr));
            all_blocks_.erase((*it)->ptr);
            it = large_blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

// ------------------- Destructor -------------------
PinnableBlockAllocator::~PinnableBlockAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &p : all_blocks_) {
        if (p.second->ptr) {
            infinirtFree(p.second->ptr);
        }
    }
    all_blocks_.clear();
    large_blocks_.clear();
    for (auto &cls : size_classes_) {
        cls.free_blocks.clear();
    }
}

} // namespace infinicore
