#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>

namespace solver::v93_energy {

using Index = std::int64_t;
using Budget = std::function<void()>;
inline constexpr std::size_t budget_batch = 4096;
struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };

struct Work {
  std::size_t columns = 0, rows = 0;
  double primal_regularization = 0;
  Index primal_terms = 0, row_terms = 0, budget_checks = 0;
  long double primal_energy = 0, row_energy = 0;
  std::optional<long double> denominator;
  const char* status = "not_started";
};

namespace implementation {

inline void budget(Work& work, const Budget& check) {
  ++work.budget_checks;
  if (check) check();
}
[[noreturn]] inline void fail(Work& work, const char* message) {
  work.status = "numerical_error";
  throw Failure(message);
}
inline long double contribution(double coefficient, double value, Work& work) {
  if (!std::isfinite(coefficient) || !std::isfinite(value))
    fail(work, "名义正则能量的系数或方向非有限");
  // 先乘系数，避免r=0或极小时无谓形成value平方的中间溢出。
  const long double result = (static_cast<long double>(coefficient) * value) * value;
  if (!std::isfinite(result) || result < 0)
    fail(work, "名义正则能量的正项超出有限范围");
  return result;
}
struct ExceptionStatus {
  Work& work;
  ~ExceptionStatus() noexcept {
    if (std::string_view(work.status) == "running") work.status = "exception";
  }
};

}  // 命名空间 implementation

// full_constant是反置换、反均衡及singleton恢复后的(vx,vz)，stored_rows为实际R存储。
// 只扫描一次，不回算旧D；不使用消元后的primal对角，避免把singleton曲率重复计入。
inline long double compute(std::size_t columns, std::span<const double> full_constant,
    std::span<const double> stored_rows, double primal_regularization,
    Work& work, const Budget& check = {}) {
  work = Work{};
  work.columns = columns; work.rows = stored_rows.size();
  work.primal_regularization = primal_regularization;
  work.status = "running";
  implementation::ExceptionStatus exception_status{work};
  implementation::budget(work, check);
  if (columns > std::numeric_limits<std::size_t>::max() - stored_rows.size() ||
      columns + stored_rows.size() != full_constant.size() ||
      columns > static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
      stored_rows.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
    implementation::fail(work, "名义正则能量的完整方向维数不一致");
  if (!(primal_regularization >= 0) || !std::isfinite(primal_regularization))
    implementation::fail(work, "名义正则能量要求有限非负原始正则");
  for (std::size_t column = 0; column < columns; ++column) {
    if ((column & (budget_batch - 1)) == 0) implementation::budget(work, check);
    const long double term = implementation::contribution(primal_regularization, full_constant[column], work);
    const long double pending = work.primal_energy + term;
    if (!std::isfinite(pending)) implementation::fail(work, "名义正则能量的原始正项累加非有限");
    work.primal_energy = pending;
    ++work.primal_terms;
  }
  for (std::size_t row = 0; row < stored_rows.size(); ++row) {
    if ((row & (budget_batch - 1)) == 0) implementation::budget(work, check);
    if (!(stored_rows[row] > 0) || !std::isfinite(stored_rows[row]))
      implementation::fail(work, "名义正则能量要求每行实际R有限正");
    const long double term = implementation::contribution(stored_rows[row], full_constant[columns + row], work);
    const long double pending = work.row_energy + term;
    if (!std::isfinite(pending)) implementation::fail(work, "名义正则能量的行正项累加非有限");
    work.row_energy = pending;
    ++work.row_terms;
  }
  const long double denominator = (1 + work.primal_energy) + work.row_energy;
  if (!(denominator >= 1) || !std::isfinite(denominator))
    implementation::fail(work, "名义正则能量分母超出有限正范围");
  // 已形成D但最后预算失败时，保留该事实，仍不算正常交付。
  work.denominator = denominator;
  implementation::budget(work, check);
  work.status = "completed";
  return denominator;
}

}  // 命名空间 solver::v93_energy
