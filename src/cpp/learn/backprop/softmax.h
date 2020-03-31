#ifndef EDGETPU_CPP_LEARN_BACKPROP_SOFTMAX_H_
#define EDGETPU_CPP_LEARN_BACKPROP_SOFTMAX_H_

#include "src/cpp/learn/backprop/operator.h"

namespace coral {
namespace learn {
namespace backprop {

// This class is a forward pass operator for the softmax classifier layer that
// computes the probibilities of each sample in the Tensor being in each class.
// A good reference for this is:
// http://cs231n.github.io/linear-classify/#softmax
class Softmax : public Operator {
 public:
  // Input: vector of size 1 of unnormalized logits; Tensor is NxC array
  // where N is number of inputs and C is number of classes.
  // Output: vector of size 1 of normalized probabilities; Tensor is NxC array.
  std::vector<Tensor> Compute(const std::vector<Tensor>& inputs) override;
};

// This class computes the gradient of the Softmax operator with respect to
// each of the elements in the vector of unnormalized logits.
// A good reference for this is: https://deepnotes.io/softmax-crossentropy
class SoftmaxGradient : public Operator {
 public:
  // Input: vector of size 2 of Tensors [logits, dprobs].
  // logits is NxC array where N is number of inputs and C is number of classes.
  // dprobs is NXC array of gradients of Loss with respect to softmax output.
  // Output: vector of size 1; Tensor is NxC gradient of Loss with respect to
  // logits.
  std::vector<Tensor> Compute(const std::vector<Tensor>& inputs) override;
};

// Helper function to compute local gradient of softmax.
Tensor softmax_local_gradient(Eigen::MatrixXf::RowXpr prob);

}  // namespace backprop
}  // namespace learn
}  // namespace coral
#endif  // EDGETPU_CPP_LEARN_BACKPROP_SOFTMAX_H_
