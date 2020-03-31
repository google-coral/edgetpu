load("//src/cpp/basic:model_benchmark_cases.bzl", "MODEL_BENCHMARK_CASES")
load("//src/cpp:model_benchmark.bzl", "generate_models_benchmark_src")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "basic_engine_native",
    srcs = [
        "basic_engine_native.cc",
    ],
    hdrs = [
        "basic_engine_native.h",
    ],
    deps = [
        ":edgetpu_resource_manager",
        "//src/cpp:error_reporter",
        "//src/cpp/posenet:posenet_decoder_op",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "basic_engine",
    srcs = [
        "basic_engine.cc",
    ],
    hdrs = [
        "basic_engine.h",
    ],
    deps = [
        ":basic_engine_native",
        "@glog",
    ],
)

cc_test(
    name = "basic_engine_native_test",
    timeout = "long",
    srcs = [
        "basic_engine_native_test.cc",
    ],
    data = [
        "//test_data:images",
        "//test_data:models",
        "//test_data/invalid_models:models",
    ],
    linkstatic = 1,
    deps = [
        ":basic_engine_native",
        "//src/cpp:fake_op",
        "//src/cpp:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "basic_engine_test",
    timeout = "long",
    srcs = [
        "basic_engine_test.cc",
    ],
    data = [
        "//test_data:images",
        "//test_data:models",
        "//test_data/invalid_models:models",
    ],
    linkstatic = 1,
    deps = [
        ":basic_engine",
        "//src/cpp:fake_op",
        "//src/cpp:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "embedding_extractor_models_test",
    timeout = "long",
    srcs = [
        "embedding_extractor_models_test.cc",
    ],
    data = [
        "//test_data:images",
        "//test_data:models",
    ],
    linkstatic = 1,
    deps = [
        ":basic_engine",
        "//src/cpp:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "segmentation_models_test",
    timeout = "long",
    srcs = [
        "segmentation_models_test.cc",
    ],
    data = [
        "//test_data:images",
        "//test_data:models",
    ],
    linkstatic = 1,
    deps = [
        ":basic_engine",
        "//src/cpp:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_library(
    name = "edgetpu_resource_manager",
    srcs = [
        "edgetpu_resource_manager.cc",
    ],
    hdrs = [
        "edgetpu_resource_manager.h",
    ],
    deps = [
        "//src/cpp:error_reporter",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@glog",
        "@libedgetpu//:header",
    ],
)

cc_test(
    name = "edgetpu_resource_manager_test",
    srcs = [
        "edgetpu_resource_manager_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":edgetpu_resource_manager",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_binary(
    name = "cocompiled_models_benchmark",
    testonly = 1,
    srcs = [
        "cocompiled_models_benchmark.cc",
    ],
    data = [
        "//test_data:models",
        "//test_data/cocompilation:models",
    ],
    deps = [
        ":basic_engine",
        "//src/cpp:test_utils",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/flags:parse",
        "@glog",
    ],
)

cc_test(
    name = "inference_repeatability_test",
    timeout = "long",
    srcs = [
        "inference_repeatability_test.cc",
    ],
    data = [
        "//test_data:models",
    ],
    linkstatic = 1,
    deps = [
        ":basic_engine",
        "//src/cpp:test_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "model_loading_stress_test",
    timeout = "long",
    srcs = [
        "model_loading_stress_test.cc",
    ],
    data = [
        "//test_data:models",
    ],
    linkstatic = 1,
    deps = [
        ":basic_engine",
        "//src/cpp:test_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "inference_stress_test",
    timeout = "long",
    srcs = [
        "inference_stress_test.cc",
    ],
    data = [
        "//test_data:models",
    ],
    linkstatic = 1,
    deps = [
        "//src/cpp:test_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_binary(
    name = "edgetpu_resource_manager_benchmark",
    testonly = 1,
    srcs = ["edgetpu_resource_manager_benchmark.cc"],
    copts = [
        "-DBENCHMARK_DISABLE_THREADS",
    ],
    deps = [
        ":edgetpu_resource_manager",
        "@com_github_google_benchmark//:benchmark",
        "@com_github_google_benchmark//:benchmark_main",
        "@glog",
        "@libedgetpu//:header",
    ],
)

cc_test(
    name = "multiple_tpus_inference_stress_test",
    srcs = [
        "multiple_tpus_inference_stress_test.cc",
    ],
    data = [
        "//test_data:images",
        "//test_data:models",
    ],
    linkstatic = 1,
    deps = [
        "//src/cpp:test_utils",
        "//src/cpp/basic:basic_engine",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "float_input_model_test",
    timeout = "long",
    srcs = [
        "float_input_model_test.cc",
    ],
    data = [
        "//test_data:images",
        "//test_data:models",
    ],
    linkstatic = 1,
    deps = [
        ":basic_engine",
        "//src/cpp:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

[generate_models_benchmark_src(
    name = "%s_src" % case.get("benchmark_name").lower(),
    benchmark_name = case.get("benchmark_name"),
    tflite_cpu_filepath = case.get("tflite_cpu_filepath"),
    tflite_edgetpu_filepath = case.get("tflite_edgetpu_filepath"),
) for case in MODEL_BENCHMARK_CASES]

cc_binary(
    name = "models_benchmark",
    testonly = 1,
    srcs = [":%s_src" % case.get("benchmark_name").lower() for case in MODEL_BENCHMARK_CASES],
    data = [
        "//test_data:models",
    ],
    deps = [
        "//src/cpp:benchmark_main",
        "//src/cpp:test_utils",
        "@com_github_google_benchmark//:benchmark",
        "@glog",
    ],
)
