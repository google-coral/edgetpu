#ifndef EDGETPU_CPP_LEARN_BACKPROP_SGD_UPDATER_H_
#define EDGETPU_CPP_LEARN_BACKPROP_SGD_UPDATER_H_

#include <vector>

#include "src/cpp/learn/backprop/tensor.h"
#include "src/cpp/learn/backprop/weight_updater.h"

namespace coral {
namespace learn {
namespace backprop {

// This class updates the weights using stochastic gradient descent.
class SgdUpdater : public WeightUpdater {
 public:
  explicit SgdUpdater(float learning_rate = .01) {
    learning_rate_ = learning_rate;
  }

  // Updates the value of weights based on grads.
  // Inputs: grads is a vector of Tensors of gradients to be used to update the
  // weights in a particular layer of a neural net, and weights is a vector of
  // Tensors of the weights that we want to update. Each element grads[i] is the
  // same size as its corresponding element weights[i].
  // When used to update a fully connected layer, the grads are dW and db from
  // the output of FullyConnectedGradient.Compute and the weights are W and b.
  void Update(const std::vector<Tensor>& grads,
              std::vector<Tensor*> weights) override;

  // Sets how fast the model learns. Can be used by the user or by future models
  // that will need a setter to vary the learning_rate_.
  void SetLearningRate(float learning_rate) { learning_rate_ = learning_rate; }

 private:
  // The learning rate is how fast the model learns; this value determines how
  // much the weights are changed based on their gradient. Future models will
  // have a varying learning_rate_.
  float learning_rate_;
};

}  // namespace backprop
}  // namespace learn
}  // namespace coral
#endif  // EDGETPU_CPP_LEARN_BACKPROP_SGD_UPDATER_H_
