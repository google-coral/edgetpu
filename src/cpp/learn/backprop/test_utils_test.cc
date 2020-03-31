#include "src/cpp/learn/backprop/test_utils.h"

#include "Eigen/Core"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace learn {
namespace backprop {
namespace {

using ::Eigen::MatrixXf;
using ::Eigen::VectorXf;
using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(TestUtilsTest, NumericalGradientOfXtimesXtranspose) {
  Tensor mat_x(1, 4);
  mat_x << 1, 2, 3, 4;
  auto x_squared = [](const Tensor& x) { return (x * x.transpose())(0, 0); };
  Tensor dx = numerical_gradient(x_squared, &mat_x);
  Tensor mat_y_expected = 2 * mat_x;
  EXPECT_THAT(dx.reshaped(),
              Pointwise(FloatNear(1e-3), mat_y_expected.reshaped()));
}

TEST(TestUtilsTest, GenerateFakeData) {
  constexpr int kNumClasses = 2;
  constexpr int kTotalNumSamples = 300;
  constexpr int kNumTraining = 200;
  constexpr int kFeatureDim = 7;
  std::vector<VectorXf> means;
  std::vector<MatrixXf> cov_mats;
  means.reserve(kNumClasses);
  cov_mats.reserve(kNumClasses);
  for (int i = 0; i < kNumClasses; ++i) {
    means.push_back(VectorXf::Random(kFeatureDim));
    cov_mats.push_back(MatrixXf::Random(kFeatureDim, kFeatureDim));
  }
  const std::vector<int> class_sizes(kNumClasses,
                                     kTotalNumSamples / kNumClasses);
  const auto fake_data =
      generate_fake_data(class_sizes, means, cov_mats, kNumTraining);
  EXPECT_EQ(fake_data.training_data.rows(), kNumTraining);
  EXPECT_EQ(fake_data.training_data.cols(), kFeatureDim);
  EXPECT_EQ(fake_data.validation_data.rows(), kTotalNumSamples - kNumTraining);
  EXPECT_EQ(fake_data.validation_data.cols(), kFeatureDim);
  EXPECT_EQ(fake_data.training_labels.size(), kNumTraining);
  EXPECT_EQ(fake_data.validation_labels.size(),
            kTotalNumSamples - kNumTraining);
}

}  // namespace
}  // namespace backprop
}  // namespace learn
}  // namespace coral
