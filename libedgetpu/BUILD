# Description:
#   Edgetpu custom op api for tflite.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "header",
    hdrs = ["edgetpu.h", "edgetpu_c.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite:context",
    ],
)
