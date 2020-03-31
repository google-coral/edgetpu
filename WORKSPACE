workspace(name = "edgetpu")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

TENSORFLOW_COMMIT = "d855adfc5a0195788bf5f92c3c7352e638aa1109"
# Command to calculate: curl -OL <FILE-URL> | sha256sum | awk '{print $1}'
TENSORFLOW_SHA256 = "b8a691dbea2bb028fa8f7ce407b70ad236dae0a8705c8010dc7bad8af7e93bac"

# Be consistent with tensorflow/WORKSPACE.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

# Be consistent with tensorflow/WORKSPACE.
http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz"],
)  # https://github.com/bazelbuild/bazel-skylib/releases

http_archive(
    name = "com_google_glog",
    sha256 = "835888ec47ee8065b3098f3ec4373717d641954970f009833ed6d466c397409a",
    strip_prefix = "glog-41f4bf9cbc3e8995d628b459f6a239df43c2b84a",
    urls = [
        "https://github.com/google/glog/archive/41f4bf9cbc3e8995d628b459f6a239df43c2b84a.tar.gz",
    ],
    build_file_content = """
licenses(['notice'])
exports_files(['CMakeLists.txt'])
load(':bazel/glog.bzl', 'glog_library')
glog_library(with_gflags=0)
""",
)

http_archive(
  name = "com_github_google_benchmark",
  sha256 = "6e40ccab16a91a7beff4b5b640b84846867e125ebce6ac0fe3a70c5bae39675f",
  strip_prefix = "benchmark-16703ff83c1ae6d53e5155df3bb3ab0bc96083be",
  urls = [
    "https://github.com/google/benchmark/archive/16703ff83c1ae6d53e5155df3bb3ab0bc96083be.tar.gz"
  ],
)

http_archive(
    name = "org_tensorflow",
    sha256 = TENSORFLOW_SHA256,
    strip_prefix = "tensorflow-" + TENSORFLOW_COMMIT,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/" + TENSORFLOW_COMMIT + ".tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(tf_repo_name = "org_tensorflow")

new_local_repository(
    name = "libedgetpu",
    path = "libedgetpu",
    build_file = "libedgetpu/BUILD"
)

new_local_repository(
    name = "glog",
    path = "third_party/glog",
    build_file = "third_party/glog/BUILD",
)

load("@org_tensorflow//third_party/py:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
new_local_repository(
    name = "python_linux",
    path = "/usr/include",
    build_file = "third_party/python/linux/BUILD",
)

new_local_repository(
    name = "python_windows",
    path = "third_party/python/windows",
    build_file = "third_party/python/windows/BUILD",
)

# Use Python from MacPorts.
new_local_repository(
    name = "python_darwin",
    path = "/opt/local/Library/Frameworks/Python.framework/Versions",
    build_file = "third_party/python/darwin/BUILD",
)

new_local_repository(
    name = "python",
    path = "third_party/python",
    build_file = "third_party/python/BUILD",
)

http_archive(
    name = "coral_crosstool",
    sha256 = "088ef98b19a45d7224be13636487e3af57b1564880b67df7be8b3b7eee4a1bfc",
    strip_prefix = "crosstool-142e930ac6bf1295ff3ba7ba2b5b6324dfb42839",
    urls = [
        "https://github.com/google-coral/crosstool/archive/142e930ac6bf1295ff3ba7ba2b5b6324dfb42839.tar.gz",
    ],
)
load("@coral_crosstool//:configure.bzl", "cc_crosstool")
cc_crosstool(name = "crosstool", additional_system_include_directories=["//include"])
