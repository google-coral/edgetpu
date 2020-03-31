#include "src/cpp/learn/backprop/cross_entropy_loss.h"

#include <vector>

#include "Eigen/src/Core/Matrix.h"

namespace coral {
namespace learn {
namespace backprop {
using Eigen::MatrixXf;

namespace {
// Computes cross entropy loss between two probability distributions c and p.
float get_loss(const Tensor& c, const Tensor& p) {
  Tensor logp = p.array().log();
  Tensor loss = -c.cwiseProduct(logp).array().rowwise().sum();
  return loss.mean();
}
}  // namespace

std::vector<Tensor> CrossEntropyLoss::Compute(
    const std::vector<Tensor>& inputs) {
  // inputs is vector of [labels, probabilities]
  return {Tensor::Constant(1, 1, get_loss(inputs[0], inputs[1]))};
}

// Gradient of loss with respect to each element ij in input p is:
// dloss/d(pij) = 1/n * -cij/pij where n is the number of rows in p.
std::vector<Tensor> CrossEntropyGradient::Compute(
    const std::vector<Tensor>& inputs) {
  const auto& c = inputs[0];
  const auto& p = inputs[1];
  return {1.0 / p.rows() * -c.array() / p.array()};
}
}  // namespace backprop
}  // namespace learn
}  // namespace coral
