#ifndef EDGETPU_CPP_LEARN_BACKPROP_OPERATOR_H_
#define EDGETPU_CPP_LEARN_BACKPROP_OPERATOR_H_

#include <vector>

#include "src/cpp/learn/backprop/tensor.h"

namespace coral {
namespace learn {
namespace backprop {

// This class performs Machine Learning operations on a vector of Tensors.
class Operator {
 public:
  virtual std::vector<Tensor> Compute(const std::vector<Tensor>& inputs) = 0;
  virtual ~Operator() {}
};

}  // namespace backprop
}  // namespace learn
}  // namespace coral

#endif  // EDGETPU_CPP_LEARN_BACKPROP_OPERATOR_H_
