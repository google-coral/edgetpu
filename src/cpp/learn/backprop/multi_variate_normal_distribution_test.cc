#include "src/cpp/learn/backprop/multi_variate_normal_distribution.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

namespace coral {
namespace learn {
namespace backprop {

typedef Eigen::MatrixXf Matrix;
typedef Eigen::VectorXf Vector;

static Matrix Covariance(const Matrix& mat) {
  Matrix centered = mat.rowwise() - mat.colwise().mean();
  Matrix cov =
      (centered.adjoint() * centered) / static_cast<float>((mat.rows() - 1));
  return cov;
}

TEST(MultiVariateNormalDistributionTest, Test) {
  Vector mean(2);
  mean << 2.0, 3.0;
  Matrix cov(2, 2);
  cov << 1, 0.3, 0.3, 0.6;
  VLOG(1) << cov;
  MultiVariateNormalDistribution dist(mean, cov);
  auto samples = dist.Sample(10000);
  auto samples_means = samples.rowwise().mean();
  EXPECT_NEAR(2.0, samples_means[0], 0.1);
  EXPECT_NEAR(3.0, samples_means[1], 0.1);
  auto samples_cov = Covariance(samples.transpose());
  VLOG(1) << "samples cov is " << samples_cov;
  EXPECT_NEAR(1.0, samples_cov(0, 0), 0.1);
  EXPECT_NEAR(0.3, samples_cov(0, 1), 0.1);
  EXPECT_NEAR(0.3, samples_cov(1, 0), 0.1);
  EXPECT_NEAR(0.6, samples_cov(1, 1), 0.1);
}

}  // namespace backprop
}  // namespace learn
}  // namespace coral
