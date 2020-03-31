package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensor",
    hdrs = ["tensor.h"],
    deps = [
        "@eigen_archive//:eigen",
    ],
)

cc_library(
    name = "operator",
    hdrs = ["operator.h"],
    deps = [
        ":tensor",
        "@eigen_archive//:eigen",
    ],
)

cc_library(
    name = "fully_connected",
    srcs = ["fully_connected.cc"],
    hdrs = ["fully_connected.h"],
    deps = [
        ":operator",
        "@eigen_archive//:eigen",
        "@glog",
    ],
)

cc_test(
    name = "fully_connected_test",
    srcs = ["fully_connected_test.cc"],
    linkstatic = 1,
    deps = [
        ":fully_connected",
        ":operator",
        ":tensor",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "softmax",
    srcs = ["softmax.cc"],
    hdrs = ["softmax.h"],
    deps = [
        ":operator",
        ":tensor",
        "@eigen_archive//:eigen",
    ],
)

cc_test(
    name = "softmax_test",
    srcs = ["softmax_test.cc"],
    linkstatic = 1,
    deps = [
        ":cross_entropy_loss",
        ":operator",
        ":softmax",
        ":tensor",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cross_entropy_loss",
    srcs = ["cross_entropy_loss.cc"],
    hdrs = ["cross_entropy_loss.h"],
    deps = [
        ":operator",
        ":tensor",
        "@eigen_archive//:eigen",
    ],
)

cc_test(
    name = "cross_entropy_loss_test",
    srcs = ["cross_entropy_loss_test.cc"],
    linkstatic = 1,
    deps = [
        ":cross_entropy_loss",
        ":operator",
        ":tensor",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_utils",
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        ":multi_variate_normal_distribution",
        ":operator",
        ":softmax_regression_model",
        ":tensor",
        "@eigen_archive//:eigen",
        "@glog",
    ],
)

cc_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.cc"],
    linkstatic = 1,
    deps = [
        ":operator",
        ":tensor",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen",
    ],
)

cc_library(
    name = "weight_updater",
    hdrs = ["weight_updater.h"],
    deps = [":tensor"],
)

cc_library(
    name = "sgd_updater",
    srcs = ["sgd_updater.cc"],
    hdrs = ["sgd_updater.h"],
    deps = [
        ":tensor",
        ":weight_updater",
    ],
)

cc_library(
    name = "softmax_regression_model",
    srcs = ["softmax_regression_model.cc"],
    hdrs = ["softmax_regression_model.h"],
    deps = [
        ":cross_entropy_loss",
        ":fully_connected",
        ":multi_variate_normal_distribution",
        ":operator",
        ":sgd_updater",
        ":softmax",
        ":tensor",
        "//src/cpp:error_reporter",
        "//src/cpp/learn:utils",
        "@eigen_archive//:eigen",
        "@glog",
    ],
)

cc_test(
    name = "sgd_updater_test",
    srcs = ["sgd_updater_test.cc"],
    linkstatic = 1,
    deps = [
        ":sgd_updater",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "softmax_regression_model_test",
    srcs = ["softmax_regression_model_test.cc"],
    linkstatic = 1,
    deps = [
        ":sgd_updater",
        ":softmax_regression_model",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen",
        "@glog",
    ],
)

cc_library(
    name = "multi_variate_normal_distribution",
    srcs = ["multi_variate_normal_distribution.cc"],
    hdrs = ["multi_variate_normal_distribution.h"],
    deps = [
        "@eigen_archive//:eigen",
        "@glog",
    ],
)

cc_test(
    name = "multi_variate_normal_distribution_test",
    srcs = ["multi_variate_normal_distribution_test.cc"],
    linkstatic = 1,
    deps = [
        ":multi_variate_normal_distribution",
        "@com_google_googletest//:gtest_main",
        "@glog",
    ],
)

cc_binary(
    name = "softmax_regression_model_benchmark",
    testonly = 1,
    srcs = [
        "softmax_regression_model_benchmark.cc",
    ],
    linkstatic = 1,
    deps = [
        ":sgd_updater",
        ":softmax_regression_model",
        ":test_utils",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/flags:parse",
        "@eigen_archive//:eigen",
        "@glog",
    ],
)
