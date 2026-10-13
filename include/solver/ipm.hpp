#pragma once

#include "solver/backend.hpp"

namespace solver {

std::string ipm_version();
SolveResult solve_ipm(const Model& model, const SolveOptions& options = {});

}  // 命名空间 solver
