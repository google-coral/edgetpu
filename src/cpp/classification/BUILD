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
    args = [
        "--test_case_csv=classification_test_cases.csv",
    ],
    data = [
        "//test_data:classification_test_cases.csv",
        "//test_data:images",
        "//test_data:models",
        "//test_data/cocompilation:models",
    ],
    linkstatic = 1,
    deps = [
        "//src/cpp:classification_model_test_main",
    ],
)
