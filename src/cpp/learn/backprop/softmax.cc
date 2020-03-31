#include "src/cpp/learn/backprop/softmax.h"

#include <vector>

namespace coral {
namespace learn {
namespace backprop {

using ::Eigen::MatrixXf;

std::vector<Tensor> Softmax::Compute(const std::vector<Tensor>& inputs) {
  Tensor exps =
      (inputs[0].array().colwise() - inputs[0].array().rowwise().maxCoeff())
          .exp();
  return {exps.array().colwise() / exps.array().rowwise().sum()};
}

// Helper function to compute the local gradient dprobs/dlogits.
// Given a single logit input prob of dimension 1XC, the local gradient is size
// CxC where C is the number of classes.
// dprobi/dlogitj = probi*(kij - probj) where probi is output of softmax at
// index i, logitj is input logit to softmax at index j, kij is kronecker_delta
// function at position ij
Tensor softmax_local_gradient(MatrixXf::RowXpr prob) {
  Tensor kronecker_delta = MatrixXf::Identity(prob.size(), prob.size());
  Tensor local = kronecker_delta.array().rowwise() - prob.array();
  local = prob.asDiagonal() * local;
  return local;
}

// Multiplies dloss/dprobs by dprobs/dlogits to output dloss/dlogits = grad.
std::vector<Tensor> SoftmaxGradient::Compute(
    const std::vector<Tensor>& inputs) {
  Softmax softmax;
  Tensor probs = softmax.Compute(inputs)[0];
  const auto& dprobs = inputs[1];
  Tensor grad = MatrixXf::Zero(probs.rows(), probs.cols());
  for (int i = 0; i < probs.rows(); i++) {
    Tensor local_grad = softmax_local_gradient(probs.row(i));
    grad.row(i) = (dprobs.row(i) * local_grad);
  }
  return {grad};
}

}  // namespace backprop
}  // namespace learn
}  // namespace coral
