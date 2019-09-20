#include "absl/flags/parse.h"
#include "benchmark/benchmark.h"
#include "glog/logging.h"
#include "src/cpp/test_utils.h"

namespace coral {

template <coral::CnnProcessorType CnnProcessor>
static void BM_MobileNetV1(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "mobilenet_v1_1.0_224_quant_edgetpu.tflite"
                              : "mobilenet_v1_1.0_224_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_MobileNetV1, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_MobileNetV1, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_MobileNetV1_25(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "mobilenet_v1_0.25_128_quant_edgetpu.tflite"
                              : "mobilenet_v1_0.25_128_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_MobileNetV1_25, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_MobileNetV1_25, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_MobileNetV1_50(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "mobilenet_v1_0.5_160_quant_edgetpu.tflite"
                              : "mobilenet_v1_0.5_160_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_MobileNetV1_50, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_MobileNetV1_50, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_MobileNetV1_75(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "mobilenet_v1_0.75_192_quant_edgetpu.tflite"
                              : "mobilenet_v1_0.75_192_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_MobileNetV1_75, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_MobileNetV1_75, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_MobileNetV1_L2Norm(benchmark::State& state) {
  const std::string model_path = coral::TestDataPath(
      (CnnProcessor == coral::kEdgeTpu)
          ? "mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite"
          : "mobilenet_v1_1.0_224_l2norm_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_MobileNetV1_L2Norm, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_MobileNetV1_L2Norm, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_MobileNetV2(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
                              : "mobilenet_v2_1.0_224_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_MobileNetV2, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_MobileNetV2, coral::kCpu);

static void BM_MobileNetV2INatPlant(benchmark::State& state) {
  coral::BenchmarkModelOnEdgeTpu(
      coral::TestDataPath(
          "mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite"),
      state);
}
BENCHMARK(BM_MobileNetV2INatPlant);

static void BM_MobileNetV2INatInsect(benchmark::State& state) {
  coral::BenchmarkModelOnEdgeTpu(
      coral::TestDataPath(
          "mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite"),
      state);
}
BENCHMARK(BM_MobileNetV2INatInsect);

static void BM_MobileNetV2INatBird(benchmark::State& state) {
  coral::BenchmarkModelOnEdgeTpu(
      coral::TestDataPath(
          "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"),
      state);
}
BENCHMARK(BM_MobileNetV2INatBird);

template <coral::CnnProcessorType CnnProcessor>
static void BM_MobileNetV1Ssd(benchmark::State& state) {
  const std::string model_path = coral::TestDataPath(
      (CnnProcessor == coral::kEdgeTpu)
          ? "mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite"
          : "mobilenet_ssd_v1_coco_quant_postprocess.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_MobileNetV1Ssd, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_MobileNetV1Ssd, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_MobileNetV2Ssd(benchmark::State& state) {
  const std::string model_path = coral::TestDataPath(
      (CnnProcessor == coral::kEdgeTpu)
          ? "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
          : "mobilenet_ssd_v2_coco_quant_postprocess.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_MobileNetV2Ssd, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_MobileNetV2Ssd, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_FaceSsd(benchmark::State& state) {
  const std::string model_path = coral::TestDataPath(
      (CnnProcessor == coral::kEdgeTpu)
          ? "mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite"
          : "mobilenet_ssd_v2_face_quant_postprocess.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_FaceSsd, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_FaceSsd, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_InceptionV1(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "inception_v1_224_quant_edgetpu.tflite"
                              : "inception_v1_224_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_InceptionV1, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_InceptionV1, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_InceptionV2(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "inception_v2_224_quant_edgetpu.tflite"
                              : "inception_v2_224_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_InceptionV2, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_InceptionV2, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_InceptionV3(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "inception_v3_299_quant_edgetpu.tflite"
                              : "inception_v3_299_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_InceptionV3, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_InceptionV3, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_InceptionV4(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "inception_v4_299_quant_edgetpu.tflite"
                              : "inception_v4_299_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_InceptionV4, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_InceptionV4, coral::kCpu);

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

template <coral::CnnProcessorType CnnProcessor>
static void BM_EfficientNetEdgeTpuSmall(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "efficientnet-edgetpu-S_quant_edgetpu.tflite"
                              : "efficientnet-edgetpu-S_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_EfficientNetEdgeTpuSmall, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_EfficientNetEdgeTpuSmall, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_EfficientNetEdgeTpuMedium(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "efficientnet-edgetpu-M_quant_edgetpu.tflite"
                              : "efficientnet-edgetpu-M_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_EfficientNetEdgeTpuMedium, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_EfficientNetEdgeTpuMedium, coral::kCpu);

template <coral::CnnProcessorType CnnProcessor>
static void BM_EfficientNetEdgeTpuLarge(benchmark::State& state) {
  const std::string model_path =
      coral::TestDataPath((CnnProcessor == coral::kEdgeTpu)
                              ? "efficientnet-edgetpu-L_quant_edgetpu.tflite"
                              : "efficientnet-edgetpu-L_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_EfficientNetEdgeTpuLarge, coral::kEdgeTpu);
BENCHMARK_TEMPLATE(BM_EfficientNetEdgeTpuLarge, coral::kCpu);

template <CnnProcessorType CnnProcessor>
static void BM_Deeplab513Mv2Dm1_WithArgMax(benchmark::State& state) {
  const std::string model_path = TestDataPath(
      (CnnProcessor == kEdgeTpu) ? "deeplabv3_mnv2_pascal_quant_edgetpu.tflite"
                                 : "deeplabv3_mnv2_pascal_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_Deeplab513Mv2Dm1_WithArgMax, kEdgeTpu);
BENCHMARK_TEMPLATE(BM_Deeplab513Mv2Dm1_WithArgMax, kCpu);

template <CnnProcessorType CnnProcessor>
static void BM_Deeplab513Mv2Dm05_WithArgMax(benchmark::State& state) {
  const std::string model_path =
      TestDataPath((CnnProcessor == kEdgeTpu)
                       ? "deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite"
                       : "deeplabv3_mnv2_dm05_pascal_quant.tflite");
  coral::BenchmarkModelOnEdgeTpu(model_path, state);
}
BENCHMARK_TEMPLATE(BM_Deeplab513Mv2Dm05_WithArgMax, kEdgeTpu);
BENCHMARK_TEMPLATE(BM_Deeplab513Mv2Dm05_WithArgMax, kCpu);

}  // namespace coral

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
