#include "absl/flags/parse.h"
#include "benchmark/benchmark.h"

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
