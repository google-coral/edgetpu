#include "src/cpp/learn/backprop/fully_connected.h"

#include <vector>

#include "Eigen/src/Core/Matrix.h"
#include "glog/logging.h"

namespace coral {
namespace learn {
namespace backprop {

std::vector<Tensor> FullyConnected::Compute(const std::vector<Tensor>& inputs) {
  VLOG(1) << "mat_x: " << inputs[0];
  VLOG(1) << "mat_w: " << inputs[1];
  VLOG(1) << "vec_b: " << inputs[2];
  Tensor mat_y = inputs[0] * inputs[1];
  mat_y.array().rowwise() += inputs[2].array()(0, Eigen::all);
  return {mat_y};
}

std::vector<Tensor> FullyConnectedGradient::Compute(
    const std::vector<Tensor>& inputs) {
  // Inputs: Tensors of [mat_x, mat_w, b, dmat_y]
  // Outputs: Tensors of [dmat_x, dmat_w, dvec_b]
  // dmat_x = dmat_y * mat_w^T
  // dmat_w = mat_x^T * dmat_y
  // dvec_b = dmat_y^T * [1]
  const auto& mat_x = inputs[0];
  const auto& mat_w = inputs[1];
  const auto& dmat_y = inputs[3];
  Tensor dmat_x = dmat_y * mat_w.transpose();
  Tensor dmat_w = mat_x.transpose() * dmat_y;
  Tensor dmat_b =
      (dmat_y.transpose() * Eigen::MatrixXf::Ones(mat_x.rows(), 1)).transpose();
  return {dmat_x, dmat_w, dmat_b};
}

}  // namespace backprop
}  // namespace learn
}  // namespace coral
