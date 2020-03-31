#ifndef EDGETPU_CPP_LEARN_BACKPROP_MULTI_VARIATE_NORMAL_DISTRIBUTION_H_
#define EDGETPU_CPP_LEARN_BACKPROP_MULTI_VARIATE_NORMAL_DISTRIBUTION_H_
#include <random>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"

namespace coral {
namespace learn {
namespace backprop {

// Multi variate normal distribution implemented with Eigen library.
class MultiVariateNormalDistribution {
 public:
  typedef Eigen::MatrixXf Matrix;
  typedef Eigen::VectorXf Vector;
  MultiVariateNormalDistribution(const Vector& mean, const Matrix& cov);

  // Samples 'num' samples from distribution.
  // Returns a [dim, num] shape matrix.
  Matrix Sample(int num);

 private:
  void Initialize();

  // Mean of the distribution.
  Vector mean_;

  // Covariance matrix of the distribution.
  Matrix cov_;

  // Multiplies this matrix with a random variable X which is drawn from
  // N(0, I) will produce a sample drawn from N(0, cov_).
  Matrix p_;

  // The dimension of the covariance matrix.
  int dim_;

  // Eigen solver which is used to compute eigen value and eigen vectors.
  Eigen::EigenSolver<Matrix> solver_;

  // Gaussian random number generator.
  std::normal_distribution<float> rand_gaussian_;
};

}  // namespace backprop
}  // namespace learn
}  // namespace coral

#endif  // EDGETPU_CPP_LEARN_BACKPROP_MULTI_VARIATE_NORMAL_DISTRIBUTION_H_
