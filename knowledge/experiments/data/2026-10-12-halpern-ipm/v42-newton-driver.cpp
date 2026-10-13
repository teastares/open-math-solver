// 仅用于冻结 v41 的独立观测；标准库头不能在匿名命名空间内引入。
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <ostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef SOLVER_VERSION
#define SOLVER_VERSION "0.5.0"
#endif

#include "instrumented.cpp"

#ifdef _WIN32
#define wmain v42_original_wmain
#else
#define main v42_original_main
#endif
#include "frozen/src/main.cpp"
#ifdef _WIN32
#undef wmain
int wmain(int argc, wchar_t* argv[]) {
  solver::v42_configure();
  return v42_original_wmain(argc, argv);
}
#else
#undef main
int main(int argc, char* argv[]) {
  solver::v42_configure();
  return v42_original_main(argc, argv);
}
#endif
