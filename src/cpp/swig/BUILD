load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

tf_py_wrap_cc(
    name = "edgetpu_cpp_wrapper",
    srcs = ["edgetpu_cpp_wrapper.i"],
    swig_includes = ["numpy.i"],
    visibility = ["//visibility:private"],
    deps = [
        ":basic_engine_python_wrapper",
        ":imprinting_engine_python_wrapper",
        "//src/cpp:version",
        "//src/cpp/learn:utils",
        "//src/cpp/learn/imprinting:engine",
        "@python",
    ],
)

cc_library(
    name = "basic_engine_python_wrapper",
    srcs = ["basic_engine_python_wrapper.cc"],
    hdrs = [
        "basic_engine_python_wrapper.h",
    ],
    deps = [
        "//src/cpp:error_reporter",
        "//src/cpp/basic:basic_engine_native",
        "@python",
    ],
)

cc_library(
    name = "imprinting_engine_python_wrapper",
    srcs = ["imprinting_engine_python_wrapper.cc"],
    hdrs = [
        "imprinting_engine_python_wrapper.h",
    ],
    deps = [
        "//src/cpp:error_reporter",
        "//src/cpp/learn/imprinting:engine_native",
        "@python",
    ],
)
