#include "src/cpp/learn/imprinting/imprinting_test_base.h"

#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/learn/utils.h"
#include "src/cpp/test_utils.h"

namespace coral {
namespace learn {
namespace imprinting {

std::string ImprintingTestBase::ImagePath(const std::string& file_name) {
  return absl::StrCat(TestDataPath("/imprinting/"), file_name);
}

std::string ImprintingTestBase::GenerateInputModelPath(
    const std::string& file_name) {
  return absl::StrCat(
      TestDataPath("/imprinting/"),
      file_name + (tpu_tflite_ ? "_edgetpu.tflite" : ".tflite"));
}

std::string ImprintingTestBase::GenerateOutputModelPath(
    const std::string& file_name) {
  return GenerateRandomFilePath(file_name,
                                (tpu_tflite_ ? "_edgetpu.tflite" : ".tflite"));
}
void ImprintingTestBase::CheckRetrainedLayers(
    const std::string& output_file_path) {
  // Checks that last 5 operators are L2Norm, Conv2d, Mul, Reshape, Softmax.
  std::string input_model_content;
  EdgeTpuErrorReporter reporter;
  EXPECT_EQ(kEdgeTpuApiOk,
            ReadFile(output_file_path, &input_model_content, &reporter));
  const tflite::Model* model = tflite::GetModel(input_model_content.data());
  const auto model_t = absl::WrapUnique<tflite::ModelT>(model->UnPack());

  auto get_builtin_opcode = [](const tflite::ModelT* model_t, int op_index) {
    auto& op = model_t->subgraphs[0]->operators[op_index];
    auto& opcodes = model_t->operator_codes;
    return opcodes[op->opcode_index]->builtin_code;
  };

  VLOG(1) << "# of operators in graph: "
          << model_t->subgraphs[0]->operators.size();

  CHECK_GE(model_t->subgraphs[0]->operators.size(), 5);
  const int last_op_index = model_t->subgraphs[0]->operators.size() - 1;
  CHECK_EQ(get_builtin_opcode(model_t.get(), last_op_index),
           tflite::BuiltinOperator_SOFTMAX);
  CHECK_EQ(get_builtin_opcode(model_t.get(), last_op_index - 1),
           tflite::BuiltinOperator_RESHAPE);
  CHECK_EQ(get_builtin_opcode(model_t.get(), last_op_index - 2),
           tflite::BuiltinOperator_MUL);
  CHECK_EQ(get_builtin_opcode(model_t.get(), last_op_index - 3),
           tflite::BuiltinOperator_CONV_2D);
  CHECK_EQ(get_builtin_opcode(model_t.get(), last_op_index - 4),
           tflite::BuiltinOperator_L2_NORMALIZATION);
}

void ImprintingTestBase::TestTrainedModel(
    const std::vector<TestDatapoint>& test_datapoints,
    const std::string& output_file_path) {
  CheckRetrainedLayers(output_file_path);

  BasicEngine basic_engine(output_file_path);
  for (const auto& test_datapoint : test_datapoints) {
    const auto& results = basic_engine.RunInference(test_datapoint.image);
    const auto& result = results[0];
    int class_max = std::distance(
        result.begin(), std::max_element(result.begin(), result.end()));
    EXPECT_EQ(test_datapoint.predicted_class_id, class_max);
    EXPECT_GT(result[class_max], test_datapoint.classification_score);
  }
}

void ImprintingTestBase::CheckMetadata(
    const std::map<int, float>& metadata_expected,
    const std::map<int, float>& metadata) {
  EXPECT_EQ(metadata_expected.size(), metadata.size());
  for (const auto entry : metadata) {
    ASSERT_THAT(metadata_expected.find(entry.first),
                ::testing::Ne(metadata_expected.end()));
    EXPECT_LT(std::abs(metadata_expected.at(entry.first) - entry.second), 0.05);
  }
}
}  // namespace imprinting
}  // namespace learn
}  // namespace coral
