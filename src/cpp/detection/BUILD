package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "engine",
    srcs = [
        "engine.cc",
    ],
    hdrs = [
        "engine.h",
    ],
    deps = [
        "//src/cpp:bbox_utils",
        "//src/cpp/basic:basic_engine",
        "@glog",
    ],
)

cc_test(
    name = "engine_test",
    srcs = [
        "engine_test.cc",
    ],
    data = [
        "//test_data:models",
    ],
    linkstatic = 1,
    deps = [
        ":engine",
        "//src/cpp:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "models_test",
    srcs = [
        "models_test.cc",
    ],
    data = [
        "//test_data:images",
        "//test_data:models",
    ],
    linkstatic = 1,
    deps = [
        ":engine",
        "//src/cpp:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)
