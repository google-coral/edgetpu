package(default_visibility = ["//visibility:public"])

cc_library(
    name = "common",
    hdrs = ["common.h"],
    deps = ["@org_tensorflow//tensorflow/lite/c:common"],
)

cc_library(
    name = "allocator",
    hdrs = ["allocator.h"],
    deps = [],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":allocator",
        ":common",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        ":allocator",
        ":common",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@glog",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "pipelined_model_runner",
    srcs = ["pipelined_model_runner.cc"],
    hdrs = ["pipelined_model_runner.h"],
    deps = [
        ":allocator",
        ":common",
        ":utils",
        "//src/cpp/pipeline/internal:default_allocator",
        "//src/cpp/pipeline/internal:memory_pool_allocator",
        "//src/cpp/pipeline/internal:segment_runner",
        "//src/cpp/pipeline/internal:thread_safe_queue",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "pipelined_model_runner_test",
    srcs = ["pipelined_model_runner_test.cc"],
    data = [
        "//test_data:models",
        "//test_data/pipeline:models",
    ],
    linkstatic = 1,
    deps = [
        ":common",
        ":pipelined_model_runner",
        ":test_utils",
        ":utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "models_test",
    srcs = ["models_test.cc"],
    data = [
        "//test_data:models",
        "//test_data/pipeline:models",
    ],
    linkstatic = 1,
    deps = [
        ":common",
        ":pipelined_model_runner",
        ":test_utils",
        ":utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)
