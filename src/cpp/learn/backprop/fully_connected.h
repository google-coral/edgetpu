#ifndef EDGETPU_CPP_LEARN_BACKPROP_FULLY_CONNECTED_H_
#define EDGETPU_CPP_LEARN_BACKPROP_FULLY_CONNECTED_H_

#include "src/cpp/learn/backprop/operator.h"

namespace coral {
namespace learn {
namespace backprop {

// This class is a forward pass operator for the fully connected layer that
// computes Y = X*W + b
// A good reference for this is: http://cs231n.github.io/linear-classify/#score
class FullyConnected : public Operator {
 public:
  // Input: vector of Tensors in order of data mat_x, weights mat_w, and bias
  // vec_b. mat_x is size NxD where N is number of inputs and D is number of
  // dimensions. mat_w is size DxC. vec_b is size 1xC.
  // Output: vector of size 1 that is layer output Y
  std::vector<Tensor> Compute(const std::vector<Tensor>& inputs) override;
};

// This class is a backward pass operator that computes gradients of the
// inputs to the fully connected layer
// A good reference for this is:
// http://cs231n.stanford.edu/2017/handouts/linear-backprop.pdf
class FullyConnectedGradient : public Operator {
 public:
  // Input: vector of Tensors in order of data mat_x, weights mat_w, bias b,
  // grad dmat_y. The Tensors mat_x, mat_w, vec_b are as described in
  // FullyConnected class, dmat_y is size NxC.
  // Output: vector of Tensors of gradients in order of dmat_x, dmat_w, dvec_b
  // and correspond in size to mat_x, mat_w, vec_b respectively
  std::vector<Tensor> Compute(const std::vector<Tensor>& inputs) override;
};

}  // namespace backprop
}  // namespace learn
}  // namespace coral
#endif  // EDGETPU_CPP_LEARN_BACKPROP_FULLY_CONNECTED_H_
