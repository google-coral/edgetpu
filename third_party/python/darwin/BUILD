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

config_setting(
    name = "py38",
    define_values = {"PY3_VER": "38"}
)

# sudo port install python35 python36 python37 python38
# sudo port install py35-numpy py36-numpy py37-numpy py38-numpy
cc_library(
  name = "python3-headers",
  hdrs = select({
    "py35": glob(["3.5/include/python3.5m/*.h",
                  "3.5/lib/python3.5/site-packages/numpy/core/include/numpy/*.h"]),
    "py36": glob(["3.6/include/python3.6m/*.h",
                  "3.6/lib/python3.6/site-packages/numpy/core/include/numpy/*.h"]),
    "py37": glob(["3.7/include/python3.7m/*.h",
                  "3.7/lib/python3.7/site-packages/numpy/core/include/numpy/*.h"]),
    "py38": glob(["3.8/include/python3.8/*.h",
                  "3.8/include/python3.8/cpython/*.h",
                  "3.8/lib/python3.8/site-packages/numpy/core/include/numpy/*.h"]),
  }, no_match_error = "PY3_VER is not specified"),
  includes = select({
    "py35": ["3.5/include/python3.5m",
             "3.5/lib/python3.5/site-packages/numpy/core/include"],
    "py36": ["3.6/include/python3.6m",
             "3.6/lib/python3.6/site-packages/numpy/core/include"],
    "py37": ["3.7/include/python3.7m",
             "3.7/lib/python3.7/site-packages/numpy/core/include"],
    "py38": ["3.8/include/python3.8",
             "3.8/include/python3.8/cpython",
             "3.8/lib/python3.8/site-packages/numpy/core/include"],
  }, no_match_error = "PY3_VER is not specified"),
  visibility = ["//visibility:public"],
)
