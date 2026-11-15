#pragma once

#include "solver/backend.hpp"

namespace solver {

std::string simplex_version();
SolveResult solve_simplex(const Model& model, const SolveOptions& options = {});

}  // 命名空间 solver
