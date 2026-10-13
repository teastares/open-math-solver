#pragma once

#include <string>

namespace solver {

struct HighsOptions {
  std::string solver = "choose";
  std::string run_crossover = "on";
  std::string presolve = "choose";
};

}  // 命名空间 solver
