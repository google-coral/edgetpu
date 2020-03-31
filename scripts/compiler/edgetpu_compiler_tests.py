"""Tests Edge TPU compiler.

Use bash ./prepare_edgetpu_compiler_test.sh to prepare Edge TPU compiler
test. Copy generated folder to target platform if necessary. Then run:

  python3 edgetpu_compiler_tests.py --test_dir [dir_name]

under GENERATED FOLDER.

Turn off CPU scaling to get reliable result for benchmarking.
  sudo cpupower frequency-set --governor performance
"""
import argparse
import csv
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

_MACHINE_TO_ARCH = {"x86_64": "k8", "aarch64": "aarch64"}

_AUTO_GEN_TEST_PREFIX = "test_auto_gen_"

# pylint: disable=line-too-long
_TFLITE_GRAPHS_MULTIPLE_MODEL_COCOMPILATION = [
    [
        {
            "out_name":
                "mobilenet_v1_0.25_128_quant_cocompiled_with_mobilenet_v1_0.5_160_quant",
            "in_name":
                "mobilenet_v1_0.25_128_quant",
        },
        {
            "out_name":
                "mobilenet_v1_0.5_160_quant_cocompiled_with_mobilenet_v1_0.25_128_quant",
            "in_name":
                "mobilenet_v1_0.5_160_quant",
        },
    ],
    [
        {
            "out_name":
                "inception_v3_299_quant_cocompiled_with_inception_v4_299_quant",
            "in_name":
                "inception_v3_299_quant",
        },
        {
            "out_name":
                "inception_v4_299_quant_cocompiled_with_inception_v3_299_quant",
            "in_name":
                "inception_v4_299_quant",
        },
    ],
    [
        {
            "out_name":
                "mobilenet_v1_0.25_128_quant_cocompiled_with_inception_v4_299_quant",
            "in_name":
                "mobilenet_v1_0.25_128_quant",
        },
        {
            "out_name":
                "inception_v4_299_quant_cocompiled_with_mobilenet_v1_0.25_128_quant",
            "in_name":
                "inception_v4_299_quant",
        },
    ],
    [
        {
            "out_name": "mobilenet_v1_1.0_224_quant_cocompiled_with_3quant",
            "in_name": "mobilenet_v1_1.0_224_quant",
        },
        {
            "out_name": "mobilenet_v1_0.25_128_quant_cocompiled_with_3quant",
            "in_name": "mobilenet_v1_0.25_128_quant",
        },
        {
            "out_name": "mobilenet_v1_0.5_160_quant_cocompiled_with_3quant",
            "in_name": "mobilenet_v1_0.5_160_quant",
        },
        {
            "out_name": "mobilenet_v1_0.75_192_quant_cocompiled_with_3quant",
            "in_name": "mobilenet_v1_0.75_192_quant",
        },
    ],
    [
        {
            "out_name": "inception_v1_224_quant_cocompiled_with_3quant",
            "in_name": "inception_v1_224_quant",
        },
        {
            "out_name": "inception_v2_224_quant_cocompiled_with_3quant",
            "in_name": "inception_v2_224_quant",
        },
        {
            "out_name": "inception_v3_299_quant_cocompiled_with_3quant",
            "in_name": "inception_v3_299_quant",
        },
        {
            "out_name": "inception_v4_299_quant_cocompiled_with_3quant",
            "in_name": "inception_v4_299_quant",
        },
    ],
    [
        {
            "out_name":
                "inception_v1_224_quant_cocompiled_with_inception_v4_299_quant",
            "in_name":
                "inception_v1_224_quant",
        },
        {
            "out_name":
                "inception_v4_299_quant_cocompiled_with_inception_v1_224_quant",
            "in_name":
                "inception_v4_299_quant",
        },
    ],
    [
        {
            "out_name":
                "mobilenet_v1_1.0_224_quant_cocompiled_with_mobilenet_v2_1.0_224_quant",
            "in_name":
                "mobilenet_v1_1.0_224_quant",
        },
        {
            "out_name":
                "mobilenet_v2_1.0_224_quant_cocompiled_with_mobilenet_v1_1.0_224_quant",
            "in_name":
                "mobilenet_v2_1.0_224_quant",
        },
    ],
]
# pylint: enable=line-too-long

# List of strings, models that should not be tested for compilation.
_SKIP_LIST = [
    "mobilenet_v1_1.0_224_l2norm_quant",
    # AutoML models are from Cloud AutoML Vision. They will not be generated
    # by the compiler, but are needed to run benchmarks.
    "automl_flowers_small_quant",
    "automl_flowers_medium_quant",
    "automl_flowers_large_quant",
]

# List of strings, models that require large amount of memory to compile. These
# models will be skipped when there is not sufficient free memory.
_HIGH_MEM_COST_LIST = [
    "vgg_16_dummy_quant",
    "vgg_19_dummy_quant",
]
_HIGH_MEM_COST_THRESHOLD_KB = 1024 * 1024  # 1G bytes


def _get_free_memory_kb():
  query_result = subprocess.check_output(
      "cat /proc/meminfo | grep MemFree", shell=True)
  free_mem_kb = int(query_result.decode("ascii").split()[1])
  print("Sytem free memory size (kb): ", free_mem_kb)
  return free_mem_kb


def _get_models_to_skip():
  models_to_skip = _SKIP_LIST
  if _get_free_memory_kb() < _HIGH_MEM_COST_THRESHOLD_KB:
    print("[WARNING]: low memory available, skip models:", _HIGH_MEM_COST_LIST)
    models_to_skip.extend(_HIGH_MEM_COST_LIST)
  return models_to_skip


def _remove_edgetpu_model(data_dir):
  """Removes edgetpu models recursively from given folder.

  Filenames with suffix `_edgetpu.tflite` will be removed.

  Args:
    data_dir: string, path to folder.
  """
  print("Removing edgetpu models from ", data_dir)
  edgetpu_model_list = glob.glob(
      os.path.join(data_dir, "**/*_edgetpu.tflite"), recursive=True)
  models_to_skip = _get_models_to_skip()
  for model_path in edgetpu_model_list:
    if models_to_skip and os.path.basename(model_path).replace(
        "_edgetpu.tflite", "") in models_to_skip:
      print("Skipping %s" % (model_path))
      continue
    print("Removing model: %s" % (os.path.join(data_dir, model_path)))
    os.remove(os.path.join(data_dir, model_path))


def _read_benchmark_numbers(benchmark_filepath):
  """Reads benchmark numbers from file.

  Args:
    benchmark_filepath: string, file path to benchmark result.

  Returns:
    A list of benchmark names and a list of corresponding performance
      number (in ns).
  """
  with open(benchmark_filepath) as csvfile:
    reader = csv.reader(csvfile)
    benchmark_numbers = [row for row in reader]
    # Get rid of information about CPU at the beginning of benchmark file.
    # Use the length of regular benchmark entry as a ruler.
    benchmark_entry_length = len(benchmark_numbers[-1])
    benchmark_numbers = [
        row for row in benchmark_numbers if len(row) == benchmark_entry_length
    ]
    # Pop up header
    benchmark_numbers.pop(0)
    names = [row[0] for row in benchmark_numbers]
    perf_numbers = [float(row[2]) for row in benchmark_numbers]
    cpu_numbers = [float(row[3]) for row in benchmark_numbers]
    return names, perf_numbers, cpu_numbers


def _symlink_test_data(src_dir, dst_dir):
  if os.path.isdir(dst_dir):
    print("Removing previous directory: ", dst_dir)
    shutil.rmtree(dst_dir)
  os.mkdir(dst_dir)
  for root, dirs, files in os.walk(src_dir):
    for name in dirs:
      new_subdir_path = os.path.join(root, name).replace(src_dir, dst_dir)
      os.mkdir(new_subdir_path)
    for name in files:
      src_path = os.path.join(root, name)
      dst_path = src_path.replace(src_dir, dst_dir, 1)
      print(src_path, "->", dst_path)
      os.symlink(src_path, dst_path)


def _find_pipeline_model_basenames(pipeline_dir):
  print(pipeline_dir)
  model_segments = glob.glob(
      os.path.join(pipeline_dir, "**/*.tflite"), recursive=True)
  print(model_segments)
  basenames = []
  for name in model_segments:
    print(name)
    m = re.match(
        os.path.join(pipeline_dir, "(?P<basename>[\w\/]+)_segment*"), name)
    print(m)
    basenames.append(m.group("basename"))
  # Remove duplicated names
  basenames = list(set(basenames))

  # Sanity check
  must_have_models = ["inception_v3_299_quant", "inception_v4_299_quant"]
  for name in must_have_models:
    assert name in basenames

  return basenames


class EdgeTpuCompilerTest(unittest.TestCase):
  """Tests Edge TPU compiler.

  Attributes:
    test_dir: string, path to generated test data.
  """

  test_dir = "/tmp/edgetpu_compiler_tests"

  @classmethod
  def generate_tests(cls):
    cls.machine = platform.machine()
    cls.arch = _MACHINE_TO_ARCH[cls.machine]
    cls.compiler_bin = os.path.join(cls.test_dir, "compiler", cls.machine,
                                    "edgetpu_compiler")
    cls.libedgetpu_path = os.path.join(cls.test_dir, "libedgetpu/direct",
                                       cls.arch)
    cls.old_data_dir = os.path.join(cls.test_dir, "old_data")
    cls.new_data_dir = os.path.join(cls.test_dir, "new_data")
    cls.pipeline_prefix = "pipeline"
    cls.pipeline_model_basenames = _find_pipeline_model_basenames(
        os.path.join(cls.old_data_dir, cls.pipeline_prefix))

    # Create symlinks in new test data folder to files in old test data files.
    _symlink_test_data(cls.old_data_dir, cls.new_data_dir)

    _remove_edgetpu_model(cls.new_data_dir)

    cls.benchmark_dir = os.path.join(cls.test_dir, "benchmark_numbers")
    if not os.path.exists(cls.benchmark_dir):
      os.mkdir(cls.benchmark_dir)
    tests_config = os.path.join(cls.test_dir,
                                "edgetpu_compiler_tests_config.txt")
    benchmarks_config = os.path.join(cls.test_dir,
                                     "edgetpu_compiler_benchmarks_config.txt")

    cls.generate_tests_from_config(tests_config)
    cls.generate_tests_from_config(benchmarks_config)

  @classmethod
  def make_test(cls, cmd):

    def run(self):
      if "benchmarks/src" in cmd:
        self.run_benchmark_test(cmd)
      else:
        self.run_correctness_test(cmd)

    return run

  @classmethod
  def generate_tests_from_config(cls, config_filepath):
    with open(config_filepath) as config_file:
      for test_cmd in config_file:
        test_cmd = test_cmd.strip()
        if test_cmd:
          test_name = test_cmd.split(" ")[0]
          test_name = test_name[test_name.find("src/") + 4:].replace("/", "_")
          test_name = "%s%s" % (_AUTO_GEN_TEST_PREFIX, test_name)
          print("Generated test name: %s" % (test_name))
          setattr(cls, test_name, cls.make_test(test_cmd))

  def run_cmd(self, cmd, print_only=False):
    print(cmd)
    env = {"LD_LIBRARY_PATH": self.libedgetpu_path}
    shell_cmd = ["%s=%s" % (key, value) for key, value in env.items()] + cmd
    print("To run the command in shell, use:\n", " ".join(shell_cmd))
    if not print_only:
      pinfo = subprocess.run(cmd, env=env)
      self.assertEqual(pinfo.returncode, 0)

  def run_correctness_test(self, cmd):
    cmd = os.path.join(self.test_dir, cmd)
    cmd = cmd.replace("{arch}", self.arch)
    cmd = cmd.replace("{test_data}", self.new_data_dir)
    cmd = cmd.replace("{pipeline_model_names}",
                      ",".join(self.pipeline_model_basenames))
    if not os.path.exists(cmd.split(" ")[0]):
      print("[WARNING]: test binary does not exist. Skip")
      return
    self.run_cmd(cmd.split())

  def run_benchmark_test(self, cmd):
    benchmark_name = cmd.split(" ")[0]
    benchmark_name = benchmark_name[benchmark_name.find("src/") + 4:].replace(
        "/", "_")
    cmd = os.path.join(self.test_dir, cmd)
    cmd = cmd.replace("{arch}", self.arch)
    if not os.path.exists(cmd.split(" ")[0]):
      print("[WARNING]: benchmark binary does not exist. Skip")
      return

    # Run against edgetpu models stored in repo.
    benchmark_old_filepath = os.path.join(
        self.benchmark_dir,
        "%s_%s_benchmark_old.csv" % (benchmark_name, self.arch))
    cmd_old_models = cmd.replace("{test_data}", self.old_data_dir)
    cmd_old_models = cmd_old_models.replace("{benchmark_result_path}",
                                            benchmark_old_filepath)
    self.run_cmd(cmd_old_models.split())
    # Run against generated models.
    benchmark_new_filepath = os.path.join(
        self.benchmark_dir,
        "%s_%s_benchmark_new.csv" % (benchmark_name, self.arch))
    cmd_new_models = cmd.replace("{test_data}", self.new_data_dir)
    cmd_new_models = cmd_new_models.replace("{benchmark_result_path}",
                                            benchmark_new_filepath)
    self.run_cmd(cmd_new_models.split())

    benchmark_diff_filepath = os.path.join(
        self.benchmark_dir,
        "%s_%s_benchmark_diff.csv" % (benchmark_name, self.arch))
    self.compare_benchmark_results(benchmark_old_filepath,
                                   benchmark_new_filepath,
                                   benchmark_diff_filepath)

  def compare_benchmark_results(self, benchmark_old_filepath,
                                benchmark_new_filepath,
                                benchmark_diff_filepath):
    old_names, old_perf_numbers, old_cpu_numbers = _read_benchmark_numbers(
        benchmark_old_filepath)
    new_names, new_perf_numbers, new_cpu_numbers = _read_benchmark_numbers(
        benchmark_new_filepath)

    with open(benchmark_diff_filepath, "w+") as diff_file:
      for i, name in enumerate(old_names):
        with self.subTest(name):
          self.assertEqual(new_names[i], name)
          new_latency_ms = new_perf_numbers[i] / 1000000.0
          new_cpu_time_ms = new_cpu_numbers[i] / 1000000.0
          old_latency_ms = old_perf_numbers[i] / 1000000.0
          old_cpu_time_ms = old_cpu_numbers[i] / 1000000.0
          diff_percent = new_latency_ms / old_latency_ms * 100
          msg = "{} \t {:.1f} ms \t (was {:.1f} ms) \t {:.2f} % ".format(
              name, new_latency_ms, old_latency_ms, diff_percent)
          print(msg)
          diff_file.write(msg + "\n")
          # Only check relative diff when absolute diff is more than 0.1 ms.
          if new_latency_ms - old_latency_ms > 0.1:
            # For models with total latency <= 2ms, allow variation up to
            # 150%.
            if old_latency_ms <= 2:
              self.assertLess(diff_percent, 150)
            # For models with CPU time > 0.85ms, allow variation up to 130%.
            elif old_cpu_time_ms > 0.85 and new_cpu_time_ms > 0.85:
              self.assertLess(diff_percent, 130)
            else:
              self.assertLess(diff_percent, 115)

  def test_single_model_compilation(self):
    print("Test compiling individual models...\n\n")
    all_models = glob.glob(
        os.path.join(self.new_data_dir, "**/*.tflite"), recursive=True)
    edgetpu_models = glob.glob(
        os.path.join(self.new_data_dir, "**/*_edgetpu.tflite"), recursive=True)
    cpu_models = [model for model in all_models if model not in edgetpu_models]
    models_to_skip = _get_models_to_skip()
    for model_path in cpu_models:
      if os.path.basename(model_path).replace(".tflite", "") in models_to_skip:
        continue
      with self.subTest(msg=model_path):
        print("Compiling model %s" % model_path)
        dst_model_path = model_path.replace(self.old_data_dir,
                                            self.new_data_dir)
        compiler_cmd = [
            self.compiler_bin,
            "--out_dir={}/".format(os.path.dirname(dst_model_path)),
            model_path,
        ]
        self.run_cmd(compiler_cmd)

  def test_multi_model_cocompilation(self):
    print("Test cocompiling models...\n\n")
    cocompilation_dir = os.path.join(self.new_data_dir, "cocompilation")
    for model_pair in _TFLITE_GRAPHS_MULTIPLE_MODEL_COCOMPILATION:
      in_names = []
      out_names = []
      for in_out_name in model_pair:
        in_names.append(in_out_name["in_name"])
        out_names.append(in_out_name["out_name"])
      with self.subTest(msg=in_names):
        print("Cocompiling models: ", in_names)
        try:
          # Generate edgetpu models in a temporary folder first as renaming is
          # needed for generated models.
          tmp_dir = tempfile.mkdtemp()
          compiler_cmd = [
              self.compiler_bin,
              "--out_dir={}/".format(tmp_dir),
          ] + [
              os.path.join(self.new_data_dir, name + ".tflite")
              for name in in_names
          ]
          self.run_cmd(compiler_cmd)
          # Move generated files to `cocompilation_dir`.
          for in_name, out_name in zip(in_names, out_names):
            src = os.path.join(tmp_dir, in_name + "_edgetpu.tflite")
            dst = os.path.join(cocompilation_dir, out_name + "_edgetpu.tflite")
            shutil.move(src, dst)
            # Keep a copy of log files as well.
            src = os.path.join(tmp_dir, in_name + "_edgetpu.log")
            dst = os.path.join(cocompilation_dir, out_name + "_edgetpu.log")
            shutil.move(src, dst)
        finally:
          shutil.rmtree(tmp_dir)

  def test_model_pipelining_compilation(self):
    print("Test model pipelining compilation...\n\n")

    for name in self.pipeline_model_basenames:
      for num_segments in [2, 3, 4]:
        with self.subTest(msg=name + "_partitioned_into_" + str(num_segments)):
          print("Compiling model %s to %d segments" % (name, num_segments))
          old_model_path = os.path.join(self.old_data_dir, name + ".tflite")
          new_pipeline_dir = os.path.dirname(
              os.path.join(self.new_data_dir, self.pipeline_prefix, name))
          compiler_cmd = [
              self.compiler_bin,
              "--out_dir={}/".format(new_pipeline_dir),
              "--num_segments={}".format(num_segments),
              old_model_path,
          ]
          self.run_cmd(compiler_cmd)


def gen_test_suite():
  """Generate test suite.

  This is needed because we want model compilation related tests to run before
  auto generated tests, as the some auto generated tests must use Edge TPU
  models generated by compiler we are testing.

  Returns:
    unittest.TestSuite
  """
  first_tests_group = []
  second_tests_group = []
  for name in EdgeTpuCompilerTest.__dict__:
    if callable(getattr(EdgeTpuCompilerTest, name)):
      if name.startswith(_AUTO_GEN_TEST_PREFIX):
        second_tests_group.append(name)
      elif name.startswith("test_"):
        first_tests_group.append(name)

  suite = unittest.TestSuite()
  for name in first_tests_group:
    print("Add test ", name)
    suite.addTest(EdgeTpuCompilerTest(name))

  for name in second_tests_group:
    print("Add test ", name)
    suite.addTest(EdgeTpuCompilerTest(name))

  return suite


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--test_dir",
      default=os.getcwd(),
      help="Directory that contains related test data.")
  args = parser.parse_args()

  EdgeTpuCompilerTest.test_dir = args.test_dir
  EdgeTpuCompilerTest.generate_tests()
  runner = unittest.TextTestRunner()
  result = runner.run(gen_test_suite())
  sys.exit(not result.wasSuccessful())
