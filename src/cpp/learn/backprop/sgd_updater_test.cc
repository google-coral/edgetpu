#include "src/cpp/learn/backprop/sgd_updater.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace learn {
namespace backprop {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::Test;

TEST(SgdUpdaterTest, UpdateWeights) {
  Tensor mat_w(2, 5);
  mat_w << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Tensor vec_b(1, 2);
  vec_b << 1, 1;

  Tensor dmat_w = mat_w / 2;
  Tensor dvec_b = vec_b / 2;
  Tensor mat_w_expected = mat_w / 2;
  Tensor vec_b_expected = vec_b / 2;
  SgdUpdater sgd_updater(1);
  std::vector<Tensor*> weights{&mat_w, &vec_b};
  sgd_updater.Update({dmat_w, dvec_b}, weights);

  EXPECT_THAT(mat_w.reshaped(),
              Pointwise(FloatNear(1e-3), mat_w_expected.reshaped()));
  EXPECT_THAT(vec_b.reshaped(),
              Pointwise(FloatNear(1e-3), vec_b_expected.reshaped()));
}

}  // namespace
}  // namespace backprop
}  // namespace learn
}  // namespace coral
