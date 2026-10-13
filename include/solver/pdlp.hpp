#pragma once

#include "solver/backend.hpp"

namespace solver {

// 只消费自主模型，数值内核不调用外部求解器或解析器。
SolveResult solve_pdlp(const Model& model, const SolveOptions& options = {});
std::string pdlp_version();

}  // 命名空间 solver
