package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "error_reporter",
    srcs = [
        "error_reporter.cc",
    ],
    hdrs = [
        "error_reporter.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite/core/api",
    ],
)

cc_test(
    name = "error_reporter_test",
    srcs = [
        "error_reporter_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":error_reporter",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "version",
    srcs = [
        "version.cc",
    ],
    hdrs = [
        "version.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@libedgetpu//:header",
    ],
)

cc_test(
    name = "version_test",
    srcs = [
        "version_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":version",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//src/cpp:error_reporter",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "bbox_utils",
    hdrs = [
        "bbox_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "bbox_utils_test",
    srcs = [
        "bbox_utils_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":bbox_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = [
        "test_utils.cc",
    ],
    hdrs = [
        "test_utils.h",
    ],
    deps = [
        ":bbox_utils",
        ":utils",
        "//src/cpp/basic:basic_engine",
        "//src/cpp/classification:engine",
        "//src/cpp/detection:engine",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
    ],
)

cc_library(
    name = "fake_op",
    testonly = 1,
    srcs = [
        "fake_op.cc",
    ],
    hdrs = [
        "fake_op.h",
    ],
    deps = [
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
    ],
)
