#ifndef EDGETPU_CPP_PIPELINE_INTERNAL_ALIGNED_ALLOC_H_
#define EDGETPU_CPP_PIPELINE_INTERNAL_ALIGNED_ALLOC_H_

#include <cstdlib>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace coral {
namespace internal {

inline void* AlignedAlloc(int alignment, size_t size) {
#if defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void* ptr;
  if (posix_memalign(&ptr, alignment, size) == 0) return ptr;
  return nullptr;
#endif
}

inline void AlignedFree(void* aligned_memory) {
#if defined(_WIN32)
  _aligned_free(aligned_memory);
#else
  free(aligned_memory);
#endif
}

}  // namespace internal
}  // namespace coral

#endif  // EDGETPU_CPP_PIPELINE_INTERNAL_ALIGNED_ALLOC_H_
