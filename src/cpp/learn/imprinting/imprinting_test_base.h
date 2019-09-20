#ifndef EDGETPU_CPP_LEARN_IMPRINTING_IMPRINTING_TEST_BASE_H_
#define EDGETPU_CPP_LEARN_IMPRINTING_IMPRINTING_TEST_BASE_H_

#include <cmath>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/test_utils.h"

namespace coral {
namespace learn {
namespace imprinting {

// Test parameters indicates whether to run tpu version of tflite model.
class ImprintingTestBase : public ::testing::TestWithParam<bool> {
 protected:
  typedef std::vector<uint8_t> Image;

  struct TestDatapoint {
    TestDatapoint(const Image& image_, const int predicted_class_id_,
                  const float classification_score_)
        : image(image_),
          predicted_class_id(predicted_class_id_),
          classification_score(classification_score_) {}
    const Image image;
    const int predicted_class_id;
    const float classification_score;
  };

  void SetUp() override { tpu_tflite_ = GetParam(); }

  std::string ImagePath(const std::string& file_name);

  std::string GenerateInputModelPath(const std::string& file_name);

  std::string GenerateOutputModelPath(const std::string& file_name);

  void CheckRetrainedLayers(const std::string& output_file_path);

  void TestTrainedModel(const std::vector<TestDatapoint>& test_datapoints,
                        const std::string& output_file_path);

  void CheckMetadata(const std::map<int, float>& metadata_expected,
                     const std::map<int, float>& metadata);

  bool tpu_tflite_ = false;
  const ImageDims target_dims_ = {{224, 224, 3}};
  const Image cat_train_0_ =
      GetInputFromImage(ImagePath("cat_train_0.bmp"), target_dims_);
  const Image hotdog_train_0_ =
      GetInputFromImage(ImagePath("hotdog_train_0.bmp"), target_dims_);
  const Image hotdog_train_1_ =
      GetInputFromImage(ImagePath("hotdog_train_1.bmp"), target_dims_);
  const Image dog_train_0_ =
      GetInputFromImage(ImagePath("dog_train_0.bmp"), target_dims_);
  const Image cat_test_0_ =
      GetInputFromImage(ImagePath("cat_test_0.bmp"), target_dims_);
  const Image hotdog_test_0_ =
      GetInputFromImage(ImagePath("hotdog_test_0.bmp"), target_dims_);
  const Image dog_test_0_ =
      GetInputFromImage(ImagePath("dog_test_0.bmp"), target_dims_);
};
}  // namespace imprinting
}  // namespace learn
}  // namespace coral

#endif  // EDGETPU_CPP_LEARN_IMPRINTING_IMPRINTING_TEST_BASE_H_
