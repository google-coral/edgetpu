#include "src/cpp/learn/backprop/multi_variate_normal_distribution.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cmath>

#include "glog/logging.h"

static std::default_random_engine generator(
    std::chrono::system_clock::now().time_since_epoch().count());

namespace coral {
namespace learn {
namespace backprop {

typedef Eigen::MatrixXf Matrix;
typedef Eigen::VectorXf Vector;

MultiVariateNormalDistribution::MultiVariateNormalDistribution(
    const Vector& mean, const Matrix& cov)
    : mean_(mean), cov_(cov), dim_(cov.rows()) {
  Initialize();
}

void MultiVariateNormalDistribution::Initialize() {
  solver_.compute(cov_, true);
  auto eigen_values = solver_.eigenvalues().real();
  VLOG(1) << "eigen_values is " << eigen_values;
  VLOG(1) << "eigen_vector is " << solver_.eigenvectors().real();
  Matrix eigen_vectors = solver_.eigenvectors().real();
  VLOG(1) << "eigen_vector is " << solver_.eigenvectors().real();
  Matrix Q = eigen_vectors;
  for (int i = 0; i < eigen_vectors.cols(); i++) {
    float norm = Q.col(i).squaredNorm();
    Q.col(i) /= norm;
  }
  VLOG(1) << "Q is " << Q;
  Vector sqrt_lambda(eigen_values.size());
  for (int i = 0; i < eigen_values.size(); i++) {
    sqrt_lambda(i) = std::sqrt(static_cast<float>(eigen_values(i)));
  }
  Matrix sqrt_lambda_matrix = sqrt_lambda.asDiagonal();
  VLOG(1) << "sqrt_lambda_matrix is " << sqrt_lambda_matrix;
  p_ = Q * sqrt_lambda_matrix;
  VLOG(1) << "P is " << p_;
}

Matrix MultiVariateNormalDistribution::Sample(int num) {
  // Initialize x;
  Matrix x(dim_, num);
  for (int i = 0; i < dim_; i++) {
    for (int j = 0; j < num; j++) {
      x(i, j) = rand_gaussian_(generator);
    }
  }
  Matrix ret = p_ * x;
  ret.colwise() += mean_;
  return ret;
}

}  // namespace backprop
}  // namespace learn
}  // namespace coral
