#ifndef EDGETPU_CPP_LEARN_BACKPROP_WEIGHT_UPDATER_H_
#define EDGETPU_CPP_LEARN_BACKPROP_WEIGHT_UPDATER_H_

#include <vector>

#include "src/cpp/learn/backprop/tensor.h"
namespace coral {
namespace learn {
namespace backprop {

// This class updates the weights of a neural net based on their gradients.
class WeightUpdater {
 public:
  virtual void Update(const std::vector<Tensor>& grads,
                      std::vector<Tensor*> weights) = 0;
  virtual ~WeightUpdater() {}
};

}  // namespace backprop
}  // namespace learn
}  // namespace coral
#endif  // EDGETPU_CPP_LEARN_BACKPROP_WEIGHT_UPDATER_H_
