// 仅供冻结源码的观测实验使用；标准库头必须位于求解器匿名命名空间外。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef SOLVER_VERSION
#define SOLVER_VERSION "0.5.0"
#endif

#include "instrumented.cpp"

#ifdef _WIN32
#define wmain v40_original_wmain
#else
#define main v40_original_main
#endif
#include "frozen/src/main.cpp"
#ifdef _WIN32
#undef wmain
int wmain(int argc, wchar_t* argv[]) {
  solver::v40_configure();
  return v40_original_wmain(argc, argv);
}
#else
#undef main
int main(int argc, char* argv[]) {
  solver::v40_configure();
  return v40_original_main(argc, argv);
}
#endif
