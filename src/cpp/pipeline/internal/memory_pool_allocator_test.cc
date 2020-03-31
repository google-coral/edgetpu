#include "src/cpp/pipeline/internal/memory_pool_allocator.h"

#include <memory>

#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace internal {
namespace {

TEST(MemoryPoolAllocator, Allocate) {
  auto allocator = absl::make_unique<MemoryPoolAllocator>(
      std::unordered_map<size_t, int>({{1024, 2}}));
  uintptr_t first_alloc = reinterpret_cast<uintptr_t>(allocator->alloc(1024));
  EXPECT_GE(first_alloc, allocator->base_addr());
  uintptr_t second_alloc = reinterpret_cast<uintptr_t>(allocator->alloc(1024));
  EXPECT_GE(second_alloc, allocator->base_addr());
  EXPECT_NE(first_alloc, second_alloc);
  void* third_alloc = allocator->alloc(1024);
  EXPECT_EQ(third_alloc, nullptr);
}

TEST(MemoryPoolAllocator, AllocateUnsupportedSize) {
  auto allocator = absl::make_unique<MemoryPoolAllocator>(
      std::unordered_map<size_t, int>({{1024, 1}}));
  auto* first_alloc = allocator->alloc(512);
  EXPECT_EQ(first_alloc, nullptr);
}

TEST(MemoryPoolAllocator, Alignment) {
  auto allocator = absl::make_unique<MemoryPoolAllocator>(
      std::unordered_map<size_t, int>({{1023, 2}}));
  uintptr_t first_alloc = reinterpret_cast<uintptr_t>(allocator->alloc(1023));
  EXPECT_EQ(first_alloc % MemoryPoolAllocator::kAlignment, 0);
  EXPECT_GE(first_alloc, allocator->base_addr());
  uintptr_t second_alloc = reinterpret_cast<uintptr_t>(allocator->alloc(1023));
  EXPECT_EQ(first_alloc % MemoryPoolAllocator::kAlignment, 0);
  EXPECT_GE(second_alloc, allocator->base_addr());
  EXPECT_NE(first_alloc, second_alloc);
}

TEST(MemoryPoolAllocator, AllocateTwoThreads) {
  auto allocator = absl::make_unique<MemoryPoolAllocator>(
      std::unordered_map<size_t, int>({{1024, 2}}));
  uintptr_t first_alloc = reinterpret_cast<uintptr_t>(allocator->alloc(1024));
  EXPECT_GE(first_alloc, allocator->base_addr());
  uintptr_t second_alloc = reinterpret_cast<uintptr_t>(allocator->alloc(1024));
  EXPECT_GE(second_alloc, allocator->base_addr());
  EXPECT_NE(first_alloc, second_alloc);
  void* third_alloc = allocator->alloc(1024);
  EXPECT_EQ(third_alloc, nullptr);
}

TEST(MemoryPoolAllocator, Deallocate) {
  auto allocator = absl::make_unique<MemoryPoolAllocator>(
      std::unordered_map<size_t, int>({{512, 1}}));
  auto* first_alloc = allocator->alloc(512);
  EXPECT_NE(first_alloc, nullptr);
  EXPECT_EQ(allocator->alloc(512), nullptr);
  allocator->free(first_alloc, 512);
  EXPECT_EQ(allocator->alloc(512), first_alloc);
}
}  // namespace
}  // namespace internal
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
