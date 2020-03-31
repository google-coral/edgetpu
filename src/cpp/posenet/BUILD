package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_test(
    name = "models_test",
    timeout = "long",
    srcs = [
        "models_test.cc",
    ],
    data = [
        "//test_data/posenet:images",
        "//test_data/posenet:models",
    ],
    linkstatic = 1,
    deps = [
        ":posenet_decoder_op",
        "//src/cpp:test_utils",
        "//src/cpp/basic:basic_engine",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_binary(
    name = "models_benchmark",
    testonly = 1,
    srcs = [
        "models_benchmark.cc",
    ],
    data = [
        "//test_data/posenet:images",
        "//test_data/posenet:models",
    ],
    deps = [
        "//src/cpp:test_utils",
        "//src/cpp/basic:basic_engine",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/flags:parse",
        "@glog",
    ],
)

cc_test(
    name = "posenet_decoder_test",
    timeout = "long",
    srcs = [
        "posenet_decoder_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":posenet_decoder",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_library(
    name = "posenet_decoder_op",
    srcs = [
        "posenet_decoder_op.cc",
    ],
    hdrs = [
        "posenet_decoder_op.h",
    ],
    deps = [
        ":posenet_decoder",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
    ],
)

cc_library(
    name = "posenet_decoder",
    srcs = [
        "posenet_decoder.cc",
    ],
    hdrs = [
        "posenet_decoder.h",
    ],
)

cc_binary(
    name = "posenet_decoder.so",
    srcs = ["posenet_decoder_tflite_plugin.cc"],
    linkopts = [
        "-Wl,-soname,posenet_decoder.so.1",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":posenet_decoder_op",
        "@org_tensorflow//tensorflow/lite:kernel_api",
        "@org_tensorflow//tensorflow/lite:util",
    ],
)
