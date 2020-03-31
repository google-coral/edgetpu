#ifndef EDGETPU_CPP_PIPELINE_INTERNAL_MEMORY_POOL_ALLOCATOR_H_
#define EDGETPU_CPP_PIPELINE_INTERNAL_MEMORY_POOL_ALLOCATOR_H_

#include <cstdint>
#include <queue>
#include <unordered_map>

#include "absl/synchronization/mutex.h"
#include "glog/logging.h"
#include "src/cpp/pipeline/allocator.h"
#include "src/cpp/pipeline/internal/aligned_alloc.h"

namespace coral {
namespace internal {

// MemoryPoolAllocator can only allocate memory of predefined sizes, with each
// predefined size allocated at most K copies. For example,
//
//   auto allocator = MemoryPoolAllocator({{1024, 4}, {512, 8}});
//
// It defines an allocator that can allocate at most 4 copies of memory of 1024
// bytes, and at most 8 copies of memory of 512 bytes.
//
// This class is thread-safe.
class MemoryPoolAllocator : public Allocator {
 public:
  // Allocated addresses are kAlignment-byte-aligned.
  static constexpr int kAlignment = 8;

  // Key is memory block size, value is number of copies.
  explicit MemoryPoolAllocator(
      const std::unordered_map<size_t, int>& size_to_copy_map);

  ~MemoryPoolAllocator() override {
    if (pool_) {
      AlignedFree(pool_);
    }
  }

  // Returns nullptr if allocation fails, either because the allocator cannot
  // handle given `size`, or allocator runs out of copies of `size`.
  void* alloc(size_t size) override {
    auto it = memory_blocks_.find(size);
    if (it == memory_blocks_.end()) {
      LOG(ERROR) << "size " << size
                 << " is not supported by MemoryPoolAllocator!";
      return nullptr;
    }
    return it->second.get();
  }

  void free(void* p, size_t size) override {
    auto it = memory_blocks_.find(size);
    if (it == memory_blocks_.end()) {
      LOG(ERROR) << "size " << size
                 << " is not supported by MemoryPoolAllocator!";
    } else {
      it->second.release(p);
    }
  }

  // Returns base address of underlying memory pool.
  uintptr_t base_addr() const { return reinterpret_cast<uintptr_t>(pool_); }

 private:
  // Defines `num_copies` copies of memory blocks of size `block_size`.
  class MemoryBlocks {
   public:
    MemoryBlocks(uintptr_t base_addr, size_t block_size, int num_copies) {
      for (int i = 0; i < num_copies; ++i) {
        blocks_.push(reinterpret_cast<void*>(base_addr + i * block_size));
      }
    }

    // Returns next available block. Returns nullptr if none available.
    void* get() {
      absl::MutexLock lock(&mu_);
      if (blocks_.empty()) {
        return nullptr;
      } else {
        void* result = blocks_.front();
        blocks_.pop();
        return result;
      }
    }

    // Releases memory block.
    void release(void* p) {
      absl::MutexLock lock(&mu_);
      blocks_.push(p);
    }

   private:
    absl::Mutex mu_;
    std::queue<void*> blocks_ ABSL_GUARDED_BY(mu_);
  };

  // Key is predifined block size, value is the corresponding memory blocks.
  std::unordered_map<size_t, MemoryBlocks> memory_blocks_;

  // Underlying memory pool.
  void* pool_;
};

}  // namespace internal
}  // namespace coral

#endif  // EDGETPU_CPP_PIPELINE_INTERNAL_MEMORY_POOL_ALLOCATOR_H_
