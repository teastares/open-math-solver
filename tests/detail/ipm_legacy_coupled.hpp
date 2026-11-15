#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace solver::v92_coupled {

using Index = std::int64_t;
using Budget = std::function<void()>;
using Preconditioner = std::function<std::vector<double>(std::span<const double>, bool refinement)>;
using Product = std::function<void(std::span<const double>, std::vector<long double>&)>;

inline constexpr std::size_t max_window = 32;
inline constexpr int max_cycles = 3;
inline constexpr std::size_t budget_batch = 4096;

struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };

// 这里只规定方程行的五块顺序；物理未知量的排列由预条件器与算子共同规定。
struct Layout {
  std::size_t columns = 0, rows = 0, inequalities = 0;
  std::size_t dimension() const {
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (inequalities > rows || columns > limit - rows ||
        columns + rows > limit - inequalities || columns + rows + inequalities > limit - 2)
      throw std::invalid_argument("联立Newton五块维数无效或溢出");
    return columns + rows + inequalities + 2;
  }
  std::size_t block(std::size_t index) const noexcept {
    const auto gap = columns + rows;
    return index < columns ? 0 : index < gap ? 1 : index == gap ? 2 :
        index < gap + 1 + inequalities ? 3 : 4;
  }
};

// 只记已启动/返回的实际调用和已完成的真验快照，不保存方向或模型引用。
struct Work {
  std::size_t dimension = 0, window = 0;
  Index preconditioner_started = 0, preconditioner_completed = 0;
  Index products_started = 0, products_completed = 0;
  Index krylov_started = 0, krylov_completed = 0, cycles_started = 0;
  Index true_checks_started = 0, true_checks_completed = 0;
  const char* status = "not_started";
  std::optional<long double> initial_measure, final_measure;
  std::array<long double, 5> block_measures{};
  std::array<double, 5> tolerances{};
};

namespace implementation {

inline void budget(const Budget& check) { if (check) check(); }
inline void coordinate_budget(std::size_t index, const Budget& check) {
  if ((index & (budget_batch - 1)) == 0) budget(check);
}
[[noreturn]] inline void arithmetic(Work& work, const char* message) {
  work.status = "numerical_error";
  throw Failure(message);
}
inline double narrow(long double value, Work& work) {
  const double result = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(result))
    arithmetic(work, "联立Newton搜索算术非有限");
  return result;
}
inline void finite(std::span<const double> values, Work& work, const Budget& check) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    coordinate_budget(i, check);
    if (!std::isfinite(values[i])) arithmetic(work, "联立Newton物理方向非有限");
  }
  budget(check);
}
inline double norm(std::span<const double> values, Work& work, const Budget& check) {
  long double scale = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    coordinate_budget(i, check);
    if (!std::isfinite(values[i])) arithmetic(work, "联立Newton搜索向量非有限");
    scale = std::max(scale, std::abs(static_cast<long double>(values[i])));
  }
  if (scale == 0) { budget(check); return 0; }
  long double squares = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    coordinate_budget(i, check);
    const long double ratio = static_cast<long double>(values[i]) / scale;
    squares += ratio * ratio;
  }
  budget(check);
  return narrow(scale * std::sqrt(squares), work);
}
inline double dot(std::span<const double> left, std::span<const double> right,
    Work& work, const Budget& check) {
  long double sum = 0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    coordinate_budget(i, check);
    sum += static_cast<long double>(left[i]) * right[i];
  }
  budget(check);
  return narrow(sum, work);
}
struct ExceptionStatus {
  Work& work;
  ~ExceptionStatus() noexcept {
    if (std::string_view(work.status) == "running") work.status = "exception";
  }
};
struct Assessment {
  std::vector<long double> residual;
  std::array<long double, 5> blocks{};
  long double measure = 0;
};

}  // 命名空间 implementation

// 初次P后直接进入有限右预条件FGMRES；没有额外IR，也不返回未达标方向。
// T是按固定五块门槛组成的对角阵，Arnoldi作用于T^-1 A P T。
// 每个周期的基点固定，重启保留全部完整真验中max-block-ratio最小的物理double点。
inline std::vector<double> solve(const Layout& layout, std::span<const double> rhs,
    const std::array<double, 5>& tolerances, const Preconditioner& precondition,
    const Product& multiply, Work& work, const Budget& check = {}) {
  work = Work{};
  work.status = "running";
  implementation::ExceptionStatus exception_status{work};
  const auto dimension = layout.dimension();
  work.dimension = dimension;
  work.window = std::min(max_window, dimension);
  work.tolerances = tolerances;
  implementation::budget(check);
  if (rhs.size() != dimension || !precondition || !multiply)
    throw std::invalid_argument("联立Newton右端、回调或维数无效");
  for (double value : tolerances)
    if (!(value > 0) || !std::isfinite(value))
      throw std::invalid_argument("联立Newton五块门槛必须有限正");
  implementation::finite(rhs, work, check);
  const auto apply = [&](std::span<const double> value, bool refinement) {
    implementation::budget(check);
    ++work.preconditioner_started;
    auto result = precondition(value, refinement);
    ++work.preconditioner_completed;
    implementation::budget(check);
    if (result.size() != dimension) throw std::invalid_argument("联立Newton预条件方向维数不一致");
    implementation::finite(result, work, check);
    return result;
  };
  const auto product = [&](std::span<const double> value, std::vector<long double>& image) {
    implementation::budget(check);
    ++work.products_started;
    multiply(value, image);
    ++work.products_completed;
    implementation::budget(check);
    if (image.size() != dimension) throw std::invalid_argument("联立Newton五块乘积维数不一致");
    for (std::size_t i = 0; i < dimension; ++i) {
      implementation::coordinate_budget(i, check);
      if (!std::isfinite(image[i])) implementation::arithmetic(work, "联立Newton真实乘积非有限");
    }
    implementation::budget(check);
  };
  const auto assess = [&](std::span<const double> direction) {
    implementation::budget(check);
    ++work.true_checks_started;
    implementation::Assessment result;
    product(direction, result.residual);
    for (std::size_t i = 0; i < dimension; ++i) {
      implementation::coordinate_budget(i, check);
      result.residual[i] = static_cast<long double>(rhs[i]) - result.residual[i];
      const auto block = layout.block(i);
      const long double ratio = std::abs(result.residual[i]) / tolerances[block];
      if (!std::isfinite(ratio)) implementation::arithmetic(work, "联立Newton真实残差非有限");
      result.blocks[block] = std::max(result.blocks[block], ratio);
      result.measure = std::max(result.measure, ratio);
    }
    implementation::budget(check);
    ++work.true_checks_completed;
    return result;
  };
  auto best = apply(rhs, false);
  auto best_assessment = assess(best);
  work.initial_measure = best_assessment.measure;
  work.final_measure = best_assessment.measure;
  work.block_measures = best_assessment.blocks;
  const auto accept = [&] {
    // 真验已过也仍需最后预算；异常不能把status提前记作严格返回。
    implementation::budget(check);
    work.status = "accepted";
  };
  if (best_assessment.measure <= 1) { accept(); return best; }
  for (int cycle = 0; cycle < max_cycles; ++cycle) {
    implementation::budget(check);
    ++work.cycles_started;
    const auto base = best;
    std::vector<std::vector<double>> basis(1, std::vector<double>(dimension));
    std::vector<std::vector<double>> directions;
    directions.reserve(work.window);
    for (std::size_t i = 0; i < dimension; ++i) {
      implementation::coordinate_budget(i, check);
      basis[0][i] = implementation::narrow(best_assessment.residual[i] / tolerances[layout.block(i)], work);
    }
    const double beta = implementation::norm(basis[0], work, check);
    if (!(beta > 0)) implementation::arithmetic(work, "联立Newton初始Krylov残差无效");
    for (std::size_t i = 0; i < dimension; ++i) {
      implementation::coordinate_budget(i, check);
      basis[0][i] /= beta;
    }
    std::vector<std::vector<double>> upper(work.window, std::vector<double>(work.window + 1));
    std::vector<double> cosine(work.window), sine(work.window), transformed(work.window + 1);
    transformed[0] = beta;
    for (std::size_t column = 0; column < work.window; ++column) {
      implementation::budget(check);
      ++work.krylov_started;
      auto physical_rhs = basis[column];
      for (std::size_t i = 0; i < dimension; ++i) {
        implementation::coordinate_budget(i, check);
        physical_rhs[i] = implementation::narrow(static_cast<long double>(physical_rhs[i]) * tolerances[layout.block(i)], work);
      }
      directions.push_back(apply(physical_rhs, true));
      std::vector<long double> image;
      product(directions.back(), image);
      std::vector<double> value(dimension);
      for (std::size_t i = 0; i < dimension; ++i) {
        implementation::coordinate_budget(i, check);
        value[i] = implementation::narrow(image[i] / tolerances[layout.block(i)], work);
      }
      // 两遍改进Gram-Schmidt，系数及物理候选的乘积用LD累加后交付double。
      for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t row = 0; row <= column; ++row) {
          const double projection = implementation::dot(basis[row], value, work, check);
          upper[column][row] = implementation::narrow(static_cast<long double>(upper[column][row]) + projection, work);
          for (std::size_t i = 0; i < dimension; ++i) {
            implementation::coordinate_budget(i, check);
            value[i] = implementation::narrow(static_cast<long double>(value[i]) -
                static_cast<long double>(projection) * basis[row][i], work);
          }
        }
      }
      const double remaining = implementation::norm(value, work, check);
      upper[column][column + 1] = remaining;
      if (remaining > 0) {
        for (std::size_t i = 0; i < dimension; ++i) {
          implementation::coordinate_budget(i, check);
          value[i] /= remaining;
        }
        basis.push_back(std::move(value));
      }
      for (std::size_t row = 0; row < column; ++row) {
        implementation::budget(check);
        const double first = upper[column][row], second = upper[column][row + 1];
        upper[column][row] = implementation::narrow(static_cast<long double>(cosine[row]) * first +
            static_cast<long double>(sine[row]) * second, work);
        upper[column][row + 1] = implementation::narrow(-static_cast<long double>(sine[row]) * first +
            static_cast<long double>(cosine[row]) * second, work);
      }
      const double diagonal = std::hypot(upper[column][column], upper[column][column + 1]);
      if (!(diagonal > 0) || !std::isfinite(diagonal)) {
        work.status = "breakdown";
        throw Failure("联立NewtonKrylov子空间退化");
      }
      cosine[column] = upper[column][column] / diagonal;
      sine[column] = upper[column][column + 1] / diagonal;
      upper[column][column] = diagonal;
      upper[column][column + 1] = 0;
      transformed[column + 1] = implementation::narrow(-static_cast<long double>(sine[column]) * transformed[column], work);
      transformed[column] = implementation::narrow(static_cast<long double>(transformed[column]) * cosine[column], work);
      implementation::budget(check);
      ++work.krylov_completed;
      if (std::abs(transformed[column + 1]) <= 1 || column + 1 == work.window || remaining == 0) {
        std::vector<double> coefficients(transformed.begin(), transformed.begin() + static_cast<std::ptrdiff_t>(column + 1));
        for (std::size_t row = column + 1; row-- > 0;) {
          implementation::budget(check);
          long double value = coefficients[row];
          for (std::size_t j = row + 1; j <= column; ++j)
            value -= static_cast<long double>(upper[j][row]) * coefficients[j];
          coefficients[row] = implementation::narrow(value / upper[row][row], work);
        }
        auto candidate = base;
        for (std::size_t i = 0; i < dimension; ++i) {
          implementation::coordinate_budget(i, check);
          long double value = base[i];
          for (std::size_t j = 0; j <= column; ++j)
            value += static_cast<long double>(coefficients[j]) * directions[j][i];
          candidate[i] = implementation::narrow(value, work);
        }
        auto checked = assess(candidate);
        if (checked.measure < best_assessment.measure) {
          best.swap(candidate);
          best_assessment = std::move(checked);
          work.final_measure = best_assessment.measure;
          work.block_measures = best_assessment.blocks;
        }
        if (best_assessment.measure <= 1) { accept(); return best; }
        // 精确happy breakdown仍须真验；未通过时只可从保留真残差有限重启。
        if (remaining == 0) break;
      }
    }
  }
  implementation::budget(check);
  work.status = "exhausted";
  throw Failure("联立NewtonFGMRES未达到完整五块精度");
}

}  // 命名空间 solver::v92_coupled
