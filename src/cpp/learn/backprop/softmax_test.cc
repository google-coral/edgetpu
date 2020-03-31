#include "src/cpp/learn/backprop/softmax.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/cpp/learn/backprop/cross_entropy_loss.h"
#include "src/cpp/learn/backprop/test_utils.h"

namespace coral {
namespace learn {
namespace backprop {
namespace {

using ::Eigen::MatrixXf;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(SoftmaxTest, OneInput) {
  Tensor logits = Tensor::Ones(1, 10);
  Tensor probs_expected = Tensor::Constant(1, 10, 0.1);
  Softmax softmax;
  std::vector<Tensor> outputs = softmax.Compute({logits});
  const auto& probs = outputs[0];
  EXPECT_THAT(probs.reshaped(),
              Pointwise(FloatEq(), probs_expected.reshaped()));
}

TEST(SoftmaxTest, TwoInputs) {
  Tensor logits = Tensor::Ones(2, 5);
  Tensor probs_expected = Tensor::Constant(2, 5, 0.2);
  Softmax softmax;
  std::vector<Tensor> outputs = softmax.Compute({logits});
  const auto& probs = outputs[0];
  EXPECT_THAT(probs.reshaped(),
              Pointwise(FloatEq(), probs_expected.reshaped()));
}

TEST(SoftmaxTest, LargeInputs) {
  Tensor logits = Tensor::Constant(2, 5, 1000000);
  Tensor probs_expected = Tensor::Constant(2, 5, 0.2);
  Softmax softmax;
  std::vector<Tensor> outputs = softmax.Compute({logits});
  const auto& probs = outputs[0];
  EXPECT_THAT(probs.reshaped(),
              Pointwise(FloatEq(), probs_expected.reshaped()));
}

// Helper function to check that the local gradient of softmax used in
// SoftmaxGradient.Compute is calculated correctly
Tensor softmax_local_gradient_naive(const Tensor& prob_row) {
  Tensor local(prob_row.size(), prob_row.size());
  Tensor kronecker = Tensor::Identity(prob_row.size(), prob_row.size());
  for (int i = 0; i < prob_row.size(); i++) {
    for (int j = 0; j < prob_row.size(); j++) {
      local(i, j) = prob_row(0, i) * (kronecker(i, j) - prob_row(0, j));
    }
  }
  return local;
}

TEST(SoftmaxGradientTest, LocalGradient) {
  Tensor test(1, 4);
  test << 7, 3, 8, 9;
  Tensor grad_naive = softmax_local_gradient_naive(test);
  Tensor grad = softmax_local_gradient(test.row(0));
  EXPECT_THAT(grad_naive.reshaped(),
              Pointwise(FloatNear(1e-3), grad.reshaped()));
}

TEST(SoftmaxGradientTest, GradientOfSummingElements) {
  Tensor logits(1, 10);
  logits << 5, 2, 8, 9, 1, -15, 0, 0, -6, 5.4;
  // dprobs is a vector of ones, because the derivative of the summed value with
  // respect to each element of probs is 1.
  Tensor dprobs = Tensor::Ones(logits.rows(), logits.cols());

  Softmax softmax;
  auto x_softmax_sum = [&softmax](const Tensor& x) {
    return softmax.Compute({x})[0].sum();
  };
  Tensor dlogits_numerical = numerical_gradient(x_softmax_sum, &logits);

  SoftmaxGradient softmax_gradient;
  std::vector<Tensor> outputs = softmax_gradient.Compute({logits, dprobs});
  const auto& dlogits = outputs[0];

  EXPECT_THAT(dlogits.reshaped(),
              Pointwise(FloatNear(1e-3), dlogits_numerical.reshaped()));
}

TEST(SoftmaxGradientTest, GradientOfSummingElementsTwoInputs) {
  Tensor logits(2, 5);
  logits << 5, 2, 8, 9, 1, -15, 0, 0, -6, 5.4;
  // dprobs is a vector of ones, because the derivative of the summed value with
  // respect to each element of probs is 1.
  Tensor dprobs = Tensor::Ones(logits.rows(), logits.cols());

  Softmax softmax;
  auto x_softmax_sum = [&softmax](const Tensor& x) {
    return softmax.Compute({x})[0].sum();
  };
  Tensor dlogits_numerical = numerical_gradient(x_softmax_sum, &logits);

  SoftmaxGradient softmax_gradient;
  std::vector<Tensor> outputs = softmax_gradient.Compute({logits, dprobs});
  const auto& dlogits = outputs[0];

  EXPECT_THAT(dlogits.reshaped(),
              Pointwise(FloatNear(1e-3), dlogits_numerical.reshaped()));
}

TEST(SoftmaxGradientTest, GradientOfCrossEntropyLoss) {
  Tensor logits(1, 10);
  logits << 5, 2, 8, 9, 1, -15, 0, 0, -6, 5.4;
  Tensor labels = MatrixXf::Zero(1, 10);
  labels(0, 4) = 1;

  Softmax softmax;
  CrossEntropyLoss cross_entropy_loss;
  auto x_softmax_cel = [&softmax, &cross_entropy_loss,
                        &labels](const Tensor& x) {
    Tensor probs = softmax.Compute({x})[0];
    return cross_entropy_loss.Compute({labels, probs})[0](0, 0);
  };
  Tensor dlogits_numerical = numerical_gradient(x_softmax_cel, &logits);

  Tensor probs = softmax.Compute({logits})[0];
  CrossEntropyGradient cross_entropy_loss_gradient;
  SoftmaxGradient softmax_gradient;
  Tensor dprobs = cross_entropy_loss_gradient.Compute({labels, probs})[0];
  std::vector<Tensor> outputs = softmax_gradient.Compute({logits, dprobs});
  const auto& dlogits = outputs[0];

  EXPECT_THAT(dlogits.reshaped(),
              Pointwise(FloatNear(1e-3), dlogits_numerical.reshaped()));
}

}  // namespace
}  // namespace backprop
}  // namespace learn
}  // namespace coral
