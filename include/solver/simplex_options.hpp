#pragma once

#include "solver/lp_optimality.hpp"

#include <cstdint>

namespace solver {

struct SimplexOptions {
  // 零目标可行化和恢复原目标后的求解共用此换基预算。
  std::int64_t max_iterations = 64;
  LpOptimalityOptions optimality;
};

}  // 命名空间 solver
