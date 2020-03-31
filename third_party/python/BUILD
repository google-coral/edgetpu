config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    }
)

config_setting(
    name = "darwin",
    values = {
        "cpu": "darwin",
    }
)

cc_library(
  name = "python",
  deps = select({
    ":windows": ["@python_windows"],
    ":darwin": ["@python_darwin//:python3-headers"],
    "//conditions:default": ["@python_linux//:python3-headers"],
  }),
  visibility = ["//visibility:public"],
)
