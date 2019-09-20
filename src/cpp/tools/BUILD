package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "tflite_graph_util",
    srcs = ["tflite_graph_util.cc"],
    hdrs = ["tflite_graph_util.h"],
    deps = [
        "//src/cpp:utils",
        "@com_google_glog//:glog",
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
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
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
        "@com_google_glog//:glog",
    ],
)
