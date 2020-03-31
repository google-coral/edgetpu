package(
    default_visibility = [
        "//src/cpp/pipeline:__subpackages__",
    ],
)

cc_library(
    name = "aligned_alloc",
    hdrs = [
        "aligned_alloc.h",
    ],
)

cc_library(
    name = "thread_safe_queue",
    hdrs = ["thread_safe_queue.h"],
    deps = [
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "default_allocator",
    hdrs = ["default_allocator.h"],
    deps = ["//src/cpp/pipeline:allocator"],
)

cc_library(
    name = "memory_pool_allocator",
    srcs = ["memory_pool_allocator.cc"],
    hdrs = ["memory_pool_allocator.h"],
    deps = [
        ":aligned_alloc",
        ":segment_runner",
        "//src/cpp/pipeline:allocator",
        "@com_google_absl//absl/synchronization",
        "@glog",
    ],
)

cc_test(
    name = "memory_pool_allocator_test",
    srcs = ["memory_pool_allocator_test.cc"],
    linkstatic = 1,
    deps = [
        ":memory_pool_allocator",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "segment_runner",
    srcs = ["segment_runner.cc"],
    hdrs = ["segment_runner.h"],
    deps = [
        ":thread_safe_queue",
        "//src/cpp/pipeline:allocator",
        "//src/cpp/pipeline:common",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "segment_runner_test",
    srcs = ["segment_runner_test.cc"],
    data = [
        "//test_data:models",
    ],
    linkstatic = 1,
    deps = [
        ":default_allocator",
        ":segment_runner",
        "//src/cpp/pipeline:test_utils",
        "//src/cpp/pipeline:utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@glog",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)
