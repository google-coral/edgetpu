#include "src/cpp/bbox_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {

TEST(BboxTest, IsBoxEmpty) {
  EXPECT_TRUE(IsBoxEmpty({0.7, 0.1, 0.5, 0.3}));
  EXPECT_TRUE(IsBoxEmpty({0.5, 0.3, 0.7, 0.1}));
  EXPECT_FALSE(IsBoxEmpty({0.5, 0.1, 0.7, 0.3}));
}

TEST(BboxTest, ComputeBoxArea) {
  EXPECT_FLOAT_EQ(0, ComputeBoxArea({0.7, 0.1, 0.5, 0.3}));
  EXPECT_FLOAT_EQ(0.06, ComputeBoxArea({0.5, 0.1, 0.7, 0.4}));
}

TEST(BboxTest, IntersectionOverUnion) {
  EXPECT_FLOAT_EQ(
      0.5, IntersectionOverUnion({0.1, 0.2, 0.5, 0.4}, {0.1, 0.2, 0.3, 0.4}));
  EXPECT_FLOAT_EQ(
      0.5, IntersectionOverUnion({0.1, 0.2, 0.5, 0.4}, {0.1, 0.2, 0.5, 0.3}));
  EXPECT_FLOAT_EQ(
      0.6, IntersectionOverUnion({0.1, 0.2, 0.5, 0.4}, {0.2, 0.2, 0.6, 0.4}));
  EXPECT_FLOAT_EQ(
      0.0, IntersectionOverUnion({0.1, 0.2, 0.5, 0.4}, {0.6, 0.2, 0.9, 0.4}));
}

TEST(DetectionCandidateTest, CompareDetectionCandidate) {
  DetectionCandidate a{BoxCornerEncoding({0.0, 0.0, 1.0, 1.0}), 1, 0.2},
      b{BoxCornerEncoding({0.0, 0.0, 1.0, 1.0}), 1, 0.5};
  // Equal.
  EXPECT_TRUE(a == DetectionCandidate(
                       {BoxCornerEncoding({0.0, 0.0, 1.0, 1.0}), 1, 0.2}));
  EXPECT_FALSE(a == DetectionCandidate(
                        {BoxCornerEncoding({0.1, 0.1, 1.0, 1.0}), 1, 0.2}));
  EXPECT_FALSE(a == DetectionCandidate(
                        {BoxCornerEncoding({0.0, 0.0, 0.9, 0.9}), 1, 0.2}));
  EXPECT_FALSE(a == DetectionCandidate(
                        {BoxCornerEncoding({0.0, 0.0, 1.0, 1.0}), 1, 0.19}));
  EXPECT_FALSE(a == DetectionCandidate(
                        {BoxCornerEncoding({0.0, 0.0, 1.0, 1.0}), 2, 0.2}));
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a == b);
  // Assign.
  DetectionCandidate tmp{BoxCornerEncoding({0.0, 0.0, 0.0, 0.0}), 5, 0.7};
  EXPECT_TRUE(a != tmp);
  tmp = a;
  EXPECT_TRUE(a == tmp);
  tmp = b;
  EXPECT_TRUE(a != tmp);
}

}  // namespace coral
