#pragma once

#include "solver/lp_optimality.hpp"

#include <cstdint>
#include <string>

namespace solver {

struct PdlpOptions {
  std::string mode = "adaptive";
  std::int64_t max_iterations = 200000;
  int check_interval = 50;
  int scaling_passes = 5;
  int history_size = 8;
  LpOptimalityOptions optimality;
};

}  // 命名空间 solver
