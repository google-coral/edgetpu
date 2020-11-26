#include "src/cpp/bbox_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {

TEST(BboxTest, IsBoxEmpty) {
  EXPECT_TRUE(IsBoxEmpty({0.7f, 0.1f, 0.5f, 0.3f}));
  EXPECT_TRUE(IsBoxEmpty({0.5f, 0.3f, 0.7f, 0.1f}));
  EXPECT_FALSE(IsBoxEmpty({0.5f, 0.1f, 0.7f, 0.3f}));
}

TEST(BboxTest, ComputeBoxArea) {
  EXPECT_FLOAT_EQ(0, ComputeBoxArea({0.7f, 0.1f, 0.5f, 0.3f}));
  EXPECT_FLOAT_EQ(0.06, ComputeBoxArea({0.5f, 0.1f, 0.7f, 0.4f}));
}

TEST(BboxTest, IntersectionOverUnion) {
  EXPECT_FLOAT_EQ(
      0.5f, IntersectionOverUnion({0.1f, 0.2f, 0.5f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}));
  EXPECT_FLOAT_EQ(
      0.5f, IntersectionOverUnion({0.1f, 0.2f, 0.5f, 0.4f}, {0.1f, 0.2f, 0.5f, 0.3f}));
  EXPECT_FLOAT_EQ(
      0.6f, IntersectionOverUnion({0.1f, 0.2f, 0.5f, 0.4f}, {0.2f, 0.2f, 0.6f, 0.4f}));
  EXPECT_FLOAT_EQ(
      0.0f, IntersectionOverUnion({0.1f, 0.2f, 0.5f, 0.4f}, {0.6f, 0.2f, 0.9f, 0.4f}));
}

TEST(DetectionCandidateTest, CompareDetectionCandidate) {
  DetectionCandidate a{BoxCornerEncoding({0.0f, 0.0f, 1.0f, 1.0f}), 1, 0.2f},
      b{BoxCornerEncoding({0.0f, 0.0f, 1.0f, 1.0f}), 1, 0.5f};
  // Equal.
  EXPECT_TRUE(a == DetectionCandidate(
                       {BoxCornerEncoding({0.0f, 0.0f, 1.0f, 1.0f}), 1, 0.2f}));
  EXPECT_FALSE(a == DetectionCandidate(
                        {BoxCornerEncoding({0.1f, 0.1f, 1.0f, 1.0f}), 1, 0.2f}));
  EXPECT_FALSE(a == DetectionCandidate(
                        {BoxCornerEncoding({0.0f, 0.0f, 0.9f, 0.9f}), 1, 0.2f}));
  EXPECT_FALSE(a == DetectionCandidate(
                        {BoxCornerEncoding({0.0f, 0.0f, 1.0f, 1.0f}), 1, 0.19f}));
  EXPECT_FALSE(a == DetectionCandidate(
                        {BoxCornerEncoding({0.0f, 0.0f, 1.0f, 1.0f}), 2, 0.2f}));
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a == b);
  // Assign.
  DetectionCandidate tmp{BoxCornerEncoding({0.0f, 0.0f, 0.0f, 0.0f}), 5, 0.7f};
  EXPECT_TRUE(a != tmp);
  tmp = a;
  EXPECT_TRUE(a == tmp);
  tmp = b;
  EXPECT_TRUE(a != tmp);
}

}  // namespace coral
