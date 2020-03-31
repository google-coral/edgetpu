config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    }
)

cc_library(
    name = "glog",
    deps = select({
        ":windows": ["//windows:glog"],
        "//conditions:default": ["@com_google_glog//:glog"],
    }),
    visibility = ["//visibility:public"],
)
