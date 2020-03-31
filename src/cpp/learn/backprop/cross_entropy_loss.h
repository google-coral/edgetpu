#ifndef EDGETPU_CPP_LEARN_BACKPROP_CROSS_ENTROPY_LOSS_H_
#define EDGETPU_CPP_LEARN_BACKPROP_CROSS_ENTROPY_LOSS_H_

#include "src/cpp/learn/backprop/operator.h"

namespace coral {
namespace learn {
namespace backprop {

// This class computes the Cross Entropy between two probability distributions
// using CE(c,p) = - sum(c*log(p)) and returns the average loss of the batch.
class CrossEntropyLoss : public Operator {
 public:
  // Inputs: vector of size 2 of [c, p]
  // Both c and p Tensors are size NxC where N is the number of distributions
  // and C is the length of each distribution.
  // When used with softmax, p is the probability output from softmax and
  // c is a batch of one-hot vectors for class labels.
  // Output: vector of size 1; Tensor is 1x1 containing the average loss value.
  std::vector<Tensor> Compute(const std::vector<Tensor>& inputs) override;
};

// This class computes the gradient of the Cross Entropy Loss with respect to
// each of the elements in probability distribution p
// A good reference for this is: https://deepnotes.io/softmax-crossentropy
class CrossEntropyGradient : public Operator {
 public:
  // Inputs: vector of size 2 of [c, p]
  // c and p described in CrossEntropyLoss class; Loss is output of the Compute
  // method in CrossEntropyLoss class.
  // Output: vector of size 1; Tensor is NxC gradient with respect to p
  std::vector<Tensor> Compute(const std::vector<Tensor>& inputs) override;
};

}  // namespace backprop
}  // namespace learn
}  // namespace coral
#endif  // EDGETPU_CPP_LEARN_BACKPROP_CROSS_ENTROPY_LOSS_H_
