#ifndef EDGETPU_CPP_PIPELINE_INTERNAL_DEFAULT_ALLOCATOR_H_
#define EDGETPU_CPP_PIPELINE_INTERNAL_DEFAULT_ALLOCATOR_H_

#include "src/cpp/pipeline/allocator.h"

namespace coral {
namespace internal {

class DefaultAllocator : public Allocator {
 public:
  DefaultAllocator() = default;
  ~DefaultAllocator() override = default;

  void* alloc(size_t size) override { return std::malloc(size); }
  void free(void* p, size_t size) override { return std::free(p); }
};

}  // namespace internal
}  // namespace coral

#endif  // EDGETPU_CPP_PIPELINE_INTERNAL_DEFAULT_ALLOCATOR_H_
