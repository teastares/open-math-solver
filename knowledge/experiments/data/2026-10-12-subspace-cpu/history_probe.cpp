#include "solver/backend.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  solver::SolveOptions options;
  options.backend = "pdlp";
  options.pdlp.mode = argv[1];
  options.time_limit = 5;
  for (int i = 2; i < argc; ++i) {
    auto result = solver::solve_file(argv[i], options);
    std::cout << std::setprecision(10) << argv[i] << ',' << options.pdlp.mode << ','
      << result.status << ',' << result.pdlp_iterations.value_or(-1) << ','
      << result.total_seconds << ',' << result.solve_seconds << ','
      << result.verification_seconds << ','
      << result.pdlp_acceleration_attempts.value_or(-1) << ','
      << result.pdlp_acceleration_accepted.value_or(-1) << ','
      << result.pdlp_operator_passes.value_or(-1) << '\n';
  }
}
