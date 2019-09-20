config_setting(
    name = "py35",
    define_values = {"PY3_VER": "35"}
)

config_setting(
    name = "py36",
    define_values = {"PY3_VER": "36"}
)

config_setting(
    name = "py37",
    define_values = {"PY3_VER": "37"}
)

cc_library(
  name = "python3-headers",
  hdrs = select({
    "py35": glob(["python3.5m/*.h",
                  "python3.5m/numpy/*.h",
                  "aarch64-linux-gnu/python3.5m/*.h",
                  "arm-linux-gnueabihf/python3.5m/*.h"]),
    "py36": glob(["python3.6m/*.h",
                  "python3.6m/numpy/*.h",
                  "aarch64-linux-gnu/python3.6m/*.h",
                  "arm-linux-gnueabihf/python3.6m/*.h"]),
    "py37": glob(["python3.7m/*.h",
                  "python3.7m/numpy/*.h",
                  "aarch64-linux-gnu/python3.7m/*.h",
                  "arm-linux-gnueabihf/python3.7m/*.h"]),
  }, no_match_error = "PY3_VER is not specified"),
  includes = select({
    "py35": [".", "python3.5m"],
    "py36": [".", "python3.6m"],
    "py37": [".", "python3.7m"],
  }, no_match_error = "PY3_VER is not specified"),
  visibility = ["//visibility:public"],
)
