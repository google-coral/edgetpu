genrule(
    name = "glog_cmake",
    srcs = [
        "@com_google_glog//:CMakeLists.txt",
        ":glog_cmake.bat"
    ],
    outs = [
        "Release/glog.dll",
        "Release/glog.lib",
        "glog/logging.h",
        "glog/raw_logging.h",
        "glog/stl_logging.h",
        "glog/vlog_is_on.h",
        "glog/log_severity.h",
    ],
    cmd = "$(location :glog_cmake.bat) $(location Release/glog.dll)"
)

cc_import(
    name = "glog_dll",
    shared_library = "Release/glog.dll",
    interface_library = "Release/glog.lib",
)

cc_library(
    name = "glog",
    hdrs = [
        "glog/logging.h",
        "glog/raw_logging.h",
        "glog/stl_logging.h",
        "glog/vlog_is_on.h",
        "glog/log_severity.h",
    ],
    deps = [
        ":glog_dll",
    ],
    includes = [
        ".",
    ],
    defines = [
        "GLOG_NO_ABBREVIATED_SEVERITIES",
    ],
    visibility = ["//visibility:public"]
)

