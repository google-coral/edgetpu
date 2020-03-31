package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "tflite_graph_util",
    srcs = ["tflite_graph_util.cc"],
    hdrs = ["tflite_graph_util.h"],
    deps = [
        "//src/cpp:utils",
        "@glog",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "join_tflite_models",
    srcs = ["join_tflite_models.cc"],
    deps = [
        ":tflite_graph_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "tflite_graph_util_test",
    srcs = ["tflite_graph_util_test.cc"],
    data = [
        "//test_data/tools:models",
    ],
    linkstatic = 1,
    deps = [
        ":tflite_graph_util",
        "//src/cpp:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_binary(
    name = "multiple_tpus_performance_analysis",
    testonly = 1,
    srcs = ["multiple_tpus_performance_analysis.cc"],
    data = [
        "//test_data:images",
        "//test_data:models",
    ],
    deps = [
        "//src/cpp:test_utils",
        "//src/cpp/basic:basic_engine",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@glog",
    ],
)

cc_binary(
    name = "model_pipelining_performance_analysis",
    testonly = 1,
    srcs = ["model_pipelining_performance_analysis.cc"],
    deps = [
        "//src/cpp/pipeline:common",
        "//src/cpp/pipeline:pipelined_model_runner",
        "//src/cpp/pipeline:test_utils",
        "//src/cpp/pipeline:utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@glog",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)
