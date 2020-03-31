#ifndef EDGETPU_CPP_LEARN_BACKPROP_TEST_UTILS_H_
#define EDGETPU_CPP_LEARN_BACKPROP_TEST_UTILS_H_

#include <functional>

#include "Eigen/Core"
#include "src/cpp/learn/backprop/operator.h"
#include "src/cpp/learn/backprop/softmax_regression_model.h"
#include "src/cpp/learn/backprop/tensor.h"

namespace coral {
namespace learn {
namespace backprop {

// Func takes in tensor and outputs a scalar value
using Func = std::function<float(const Tensor&)>;

// Gets numerical gradient of f at point x to use in backprop gradient checking
// uses dx = f(x+h)-f(x-h)/(2*h) as numerical approximation where h is epsilon
Tensor numerical_gradient(Func f, Tensor* x, float epsilon = 1e-3);

// Helper function to generate data in which examples from the same class are
// drawn from the same MultiVariate Normal (MVN) Distribution.
// Note that this function generates real random values. If this leads to
// flakiness consider change it to create same pseudo random value sequence.
TrainingData generate_fake_data(const std::vector<int>& class_sizes,
                                const std::vector<Eigen::VectorXf>& means,
                                const std::vector<Eigen::MatrixXf>& cov_mats,
                                int num_train);

}  // namespace backprop
}  // namespace learn
}  // namespace coral
#endif  // EDGETPU_CPP_LEARN_BACKPROP_TEST_UTILS_H_
