#include "src/cpp/learn/backprop/fully_connected.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/cpp/learn/backprop/test_utils.h"

namespace coral {
namespace learn {
namespace backprop {
namespace {

using ::Eigen::MatrixXf;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(FullyConnectedTest, SimpleInputs) {
  Tensor mat_x(2, 5);
  mat_x << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Tensor mat_w = mat_x.transpose();
  Tensor vec_b(1, 2);
  vec_b << 1, 1;
  std::vector<Tensor> inputs = {mat_x, mat_w, vec_b};
  Tensor mat_y_expected(2, 2);
  mat_y_expected << 31, 81, 81, 256;
  FullyConnected fc_forward;
  std::vector<Tensor> outputs = fc_forward.Compute(inputs);
  Tensor* mat_y = &outputs[0];
  EXPECT_THAT(mat_y->reshaped(),
              Pointwise(FloatEq(), mat_y_expected.reshaped()));
}

TEST(FullyConnectedGradientTest, GradientOfAveragingElementsOfY) {
  Tensor mat_x(2, 3);
  mat_x << 0, 1, 2, 3, 4, 5;
  Tensor mat_w(3, 1);
  mat_w << 0, 1, 2;
  Tensor vec_b(1, 1);
  vec_b << 7;
  std::vector<Tensor> inputs = {mat_x, mat_w, vec_b};

  // dmat_y is a vector of constants equal to 1/numElements, because the
  // derivative of the Average value with respect to each element of mat_y is
  // 1/numElements
  Tensor dmat_y = Tensor::Ones(mat_x.rows(), mat_w.cols());
  dmat_y *= 1.0 / dmat_y.size();

  FullyConnected fc;

  auto x_fc_avg = [&fc, &mat_w, &vec_b](const Tensor& x) {
    return fc.Compute({x, mat_w, vec_b})[0].mean();
  };
  auto w_fc_avg = [&fc, &mat_x, &vec_b](const Tensor& w) {
    return fc.Compute({mat_x, w, vec_b})[0].mean();
  };
  auto b_fc_avg = [&fc, &mat_x, &mat_w](const Tensor& b) {
    return fc.Compute({mat_x, mat_w, b})[0].mean();
  };
  Tensor dx_numerical = numerical_gradient(x_fc_avg, &mat_x);
  Tensor dw_numerical = numerical_gradient(w_fc_avg, &mat_w);
  Tensor db_numerical = numerical_gradient(b_fc_avg, &vec_b);

  inputs.push_back(dmat_y);
  FullyConnectedGradient fc_gradient;
  std::vector<Tensor> outputs = fc_gradient.Compute(inputs);
  const auto& dmat_x = outputs[0];
  const auto& dmat_w = outputs[1];
  const auto& dvec_b = outputs[2];

  EXPECT_THAT(dmat_x.reshaped(),
              Pointwise(FloatNear(2e-3), dx_numerical.reshaped()));
  EXPECT_THAT(dmat_w.reshaped(),
              Pointwise(FloatNear(2e-3), dw_numerical.reshaped()));
  EXPECT_THAT(dvec_b.reshaped(),
              Pointwise(FloatNear(2e-3), db_numerical.reshaped()));
}

TEST(FullyConnectedGradientTest, GradientOfSummingElementsOfY) {
  Tensor mat_x(3, 3);
  mat_x << 0, 1, 2, 3, 4, 5, 6, 7, 8;
  Tensor mat_w(3, 5);
  mat_w << 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2;
  Tensor vec_b(1, 5);
  vec_b << 4, 3, 2, 1, 0;

  // dmat_y is a vector of ones, because the derivative of the Summed value with
  // respect to each element of mat_y is 1.
  Tensor dmat_y = MatrixXf::Ones(mat_x.rows(), mat_w.cols());

  FullyConnected fc;
  auto x_fc_sum = [&fc, &mat_w, &vec_b](const Tensor& x) {
    return fc.Compute({x, mat_w, vec_b})[0].sum();
  };
  auto w_fc_sum = [&fc, &mat_x, &vec_b](const Tensor& w) {
    return fc.Compute({mat_x, w, vec_b})[0].sum();
  };
  auto b_fc_sum = [&fc, &mat_x, &mat_w](const Tensor& b) {
    return fc.Compute({mat_x, mat_w, b})[0].sum();
  };
  Tensor dx_numerical = numerical_gradient(x_fc_sum, &mat_x);
  Tensor dw_numerical = numerical_gradient(w_fc_sum, &mat_w);
  Tensor db_numerical = numerical_gradient(b_fc_sum, &vec_b);

  FullyConnectedGradient fc_gradient;
  std::vector<Tensor> outputs =
      fc_gradient.Compute({mat_x, mat_w, vec_b, dmat_y});
  const auto& dmat_x = outputs[0];
  const auto& dmat_w = outputs[1];
  const auto& dvec_b = outputs[2];

  EXPECT_THAT(dmat_x.reshaped(),
              Pointwise(FloatNear(5e-2), dx_numerical.reshaped()));
  EXPECT_THAT(dmat_w.reshaped(),
              Pointwise(FloatNear(5e-2), dw_numerical.reshaped()));
  EXPECT_THAT(dvec_b.reshaped(),
              Pointwise(FloatNear(5e-2), db_numerical.reshaped()));
}

}  // namespace
}  // namespace backprop
}  // namespace learn
}  // namespace coral
