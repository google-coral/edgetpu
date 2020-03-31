#include "absl/flags/parse.h"
#include "benchmark/benchmark.h"
#include "glog/logging.h"
#include "src/cpp/test_utils.h"

namespace coral {

template <CompilationType Compilation>
static void BM_Compilation_TwoSmall(benchmark::State& state) {
  const std::string model_path0 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/mobilenet_v1_0.25_128_quant_cocompiled_with_"
            "mobilenet_v1_0.5_160_quant_edgetpu.tflite"
          : "mobilenet_v1_0.25_128_quant_edgetpu.tflite");
  const std::string model_path1 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/mobilenet_v1_0.5_160_quant_cocompiled_with_"
            "mobilenet_v1_0.25_128_quant_edgetpu.tflite"
          : "mobilenet_v1_0.5_160_quant_edgetpu.tflite");
  const std::vector<std::string> model_paths = {model_path0, model_path1};
  coral::BenchmarkModelsOnEdgeTpu(model_paths, state);
}
BENCHMARK_TEMPLATE(BM_Compilation_TwoSmall, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_TwoSmall, kSingleCompilation);

template <CompilationType Compilation>
static void BM_Compilation_TwoLarge(benchmark::State& state) {
  const std::string model_path0 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/inception_v4_299_quant_cocompiled_with_inception_v3_"
            "299_quant_edgetpu.tflite"
          : "inception_v4_299_quant_edgetpu.tflite");
  const std::string model_path1 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/inception_v3_299_quant_cocompiled_with_inception_v4_"
            "299_quant_edgetpu.tflite"
          : "inception_v3_299_quant_edgetpu.tflite");
  const std::vector<std::string> model_paths = {model_path0, model_path1};
  coral::BenchmarkModelsOnEdgeTpu(model_paths, state);
}
BENCHMARK_TEMPLATE(BM_Compilation_TwoLarge, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_TwoLarge, kSingleCompilation);

template <CompilationType Compilation>
static void BM_Compilation_Small_Large(benchmark::State& state) {
  const std::string model_path0 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/mobilenet_v1_0.25_128_quant_cocompiled_with_"
            "inception_v4_299_quant_edgetpu.tflite"
          : "mobilenet_v1_0.25_128_quant_edgetpu.tflite");
  const std::string model_path1 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/inception_v4_299_quant_cocompiled_with_mobilenet_v1_"
            "0.25_128_quant_edgetpu.tflite"
          : "inception_v4_299_quant_edgetpu.tflite");
  const std::vector<std::string> model_paths = {model_path0, model_path1};
  coral::BenchmarkModelsOnEdgeTpu(model_paths, state);
}
BENCHMARK_TEMPLATE(BM_Compilation_Small_Large, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_Small_Large, kSingleCompilation);

template <CompilationType Compilation>
static void BM_Compilation_Large_Small(benchmark::State& state) {
  const std::string model_path0 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/inception_v4_299_quant_cocompiled_with_mobilenet_v1_"
            "0.25_128_quant_edgetpu.tflite"
          : "inception_v4_299_quant_edgetpu.tflite");
  const std::string model_path1 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/mobilenet_v1_0.25_128_quant_cocompiled_with_"
            "inception_v4_299_quant_edgetpu.tflite"
          : "mobilenet_v1_0.25_128_quant_edgetpu.tflite");
  const std::vector<std::string> model_paths = {model_path0, model_path1};
  coral::BenchmarkModelsOnEdgeTpu(model_paths, state);
}
BENCHMARK_TEMPLATE(BM_Compilation_Large_Small, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_Large_Small, kSingleCompilation);

template <CompilationType Compilation>
static void BM_Compilation_Four_Small(benchmark::State& state) {
  const std::string model_path0 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "mobilenet_v1_1.0_224_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "mobilenet_v1_1.0_224_quant_edgetpu.tflite");
  const std::string model_path1 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "mobilenet_v1_0.25_128_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "mobilenet_v1_0.25_128_quant_edgetpu.tflite");
  const std::string model_path2 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "mobilenet_v1_0.5_160_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "mobilenet_v1_0.5_160_quant_edgetpu.tflite");
  const std::string model_path3 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "mobilenet_v1_0.75_192_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "mobilenet_v1_0.75_192_quant_edgetpu.tflite");
  const std::vector<std::string> model_paths = {model_path0, model_path1,
                                                model_path2, model_path3};
  coral::BenchmarkModelsOnEdgeTpu(model_paths, state);
}
BENCHMARK_TEMPLATE(BM_Compilation_Four_Small, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_Four_Small, kSingleCompilation);

template <CompilationType Compilation>
static void BM_Compilation_Four_Large(benchmark::State& state) {
  const std::string model_path0 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "inception_v1_224_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "inception_v1_224_quant_edgetpu.tflite");
  const std::string model_path1 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "inception_v2_224_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "inception_v2_224_quant_edgetpu.tflite");
  const std::string model_path2 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "inception_v3_299_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "inception_v3_299_quant_edgetpu.tflite");
  const std::string model_path3 = TestDataPath(
      (Compilation == kCoCompilation)
          ? "cocompilation/"
            "inception_v4_299_quant_cocompiled_with_3quant_edgetpu.tflite"
          : "inception_v4_299_quant_edgetpu.tflite");
  const std::vector<std::string> model_paths = {model_path0, model_path1,
                                                model_path2, model_path3};
  coral::BenchmarkModelsOnEdgeTpu(model_paths, state);
}
BENCHMARK_TEMPLATE(BM_Compilation_Four_Large, kCoCompilation);
BENCHMARK_TEMPLATE(BM_Compilation_Four_Large, kSingleCompilation);

}  // namespace coral

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
