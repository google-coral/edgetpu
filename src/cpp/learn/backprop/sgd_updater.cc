#include "src/cpp/learn/backprop/sgd_updater.h"

namespace coral {
namespace learn {
namespace backprop {

void SgdUpdater::Update(const std::vector<Tensor>& grads,
                        std::vector<Tensor*> weights) {
  for (int i = 0; i < weights.size(); i++) {
    *(weights[i]) -= learning_rate_ * grads[i];
  }
}
}  // namespace backprop
}  // namespace learn
}  // namespace coral
