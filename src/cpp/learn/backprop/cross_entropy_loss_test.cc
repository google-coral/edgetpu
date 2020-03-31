#include "src/cpp/learn/backprop/cross_entropy_loss.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/cpp/learn/backprop/test_utils.h"

namespace coral {
namespace learn {
namespace backprop {
namespace {

using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(CrossEntropyLossTest, OneInput) {
  Tensor probs = Tensor::Constant(1, 10, 0.1);
  Tensor labels = Eigen::MatrixXf::Zero(1, 10);
  labels(0, 4) = 1;
  Tensor loss_expected(1, 1);
  loss_expected << -std::log(.1);
  CrossEntropyLoss cross_entropy_loss;
  std::vector<Tensor> outputs = cross_entropy_loss.Compute({labels, probs});
  const auto& loss = outputs[0];
  EXPECT_THAT(loss.reshaped(), Pointwise(FloatEq(), loss_expected.reshaped()));
}

TEST(CrossEntropyLossTest, TwoInputs) {
  Tensor probs(2, 10);
  probs << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.02, 0.01,
      0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01;
  Tensor labels = Eigen::MatrixXf::Zero(2, 10);
  labels(0, 4) = 1;
  labels(1, 0) = 1;
  Tensor loss_expected(1, 1);
  loss_expected << -1 * (std::log(.1) + std::log(.9)) / probs.rows();
  CrossEntropyLoss cross_entropy_loss;
  std::vector<Tensor> outputs = cross_entropy_loss.Compute({labels, probs});
  const auto& loss = outputs[0];
  EXPECT_THAT(loss.reshaped(), Pointwise(FloatEq(), loss_expected.reshaped()));
}

TEST(CrossEntropyLossGradientTest, GradientofLoss) {
  Tensor probs(2, 10);
  probs << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.02, 0.01,
      0.01, 0.01, 0.01, 0.01, 0.9, 0.01, 0.01;
  Tensor labels = Eigen::MatrixXf::Zero(2, 10);
  labels(0, 4) = 1;
  labels(1, 7) = 1;
  CrossEntropyLoss cross_entropy_loss;
  auto calculate_loss = [&cross_entropy_loss, &labels](const Tensor& x) {
    return cross_entropy_loss.Compute({labels, x})[0](0, 0);
  };
  Tensor dprobs_numerical = numerical_gradient(calculate_loss, &probs);

  CrossEntropyGradient cross_entropy_loss_gradient;
  std::vector<Tensor> outputs =
      cross_entropy_loss_gradient.Compute({labels, probs});
  const auto& dprobs = outputs[0];

  EXPECT_THAT(dprobs.reshaped(),
              Pointwise(FloatNear(1e-3), dprobs_numerical.reshaped()));
}

}  // namespace
}  // namespace backprop
}  // namespace learn
}  // namespace coral
