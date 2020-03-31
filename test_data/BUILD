package(default_visibility = ["//visibility:public"])

exports_files(
    [
        "efficientnet-edgetpu-L_quant.tflite",
        "efficientnet-edgetpu-M_quant.tflite",
        "efficientnet-edgetpu-S_quant.tflite",
        "classification_test_cases.csv",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "models",
    srcs = glob(["*.tflite"]),
)

filegroup(
    name = "labels",
    srcs = glob(["*.txt"]),
)

filegroup(
    name = "images",
    srcs = glob([
        "*.bmp",
        "*.jpg",
    ]),
)
