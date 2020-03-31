#ifndef EDGETPU_CPP_PIPELINE_ALLOCATOR_H_
#define EDGETPU_CPP_PIPELINE_ALLOCATOR_H_

#include <cstddef>
#include <cstdlib>

namespace coral {

// Memory allocator used by PipelinedModelRunner to allocate input and output
// tensors.
class Allocator {
 public:
  Allocator() = default;
  virtual ~Allocator() = default;
  Allocator(const Allocator&) = delete;
  Allocator& operator=(const Allocator&) = delete;

  // Allocates `size` bytes of memory.
  // @param size The number of bytes to allocate.
  // @return A pointer to the memory, or nullptr if allocation fails.
  virtual void* alloc(size_t size) = 0;
  // Deallocates memory at the given block.
  // @param p A pointer to the memory to deallocate.
  // @param size NOT USED by the default allocator.
  virtual void free(void* p, size_t size) = 0;
};
}  // namespace coral

#endif  // EDGETPU_CPP_PIPELINE_ALLOCATOR_H_
