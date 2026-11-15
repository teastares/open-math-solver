#include "solver/detail/ipm_coupled_state.hpp"
#include "solver/detail/normal_cholesky.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Index = std::int64_t;
using Clock = std::chrono::steady_clock;
using Factor = solver::detail::NormalCholesky;
constexpr int epochs = 8, warmup = 3, blocks = 9;
constexpr long double tolerance = 1e-11L;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
Index ns(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

// 确定的64位字滚动摘要，只作一致性证据；不替代逐位组件检查或数学验收。
struct Digest {
  std::uint64_t value = 14695981039346656037ULL;
  void add(std::uint64_t word) noexcept { value ^= word; value *= 1099511628211ULL; }
  void add(double number) noexcept { add(std::bit_cast<std::uint64_t>(number)); }
  std::string text() const {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << value;
    return stream.str();
  }
};
std::uint64_t contents(std::span<const double> values) {
  Digest result;
  result.add(static_cast<std::uint64_t>(values.size()));
  for (double value : values) result.add(value);
  return result.value;
}

struct Layout {
  std::size_t n;
  int width;
  std::string structure;
  std::vector<Index> starts{0}, rows;
  std::vector<std::int8_t> signs;
  std::uint64_t contributions = 0;
  Layout(std::size_t size, int degree, const std::string& name) : n(size), width(degree), structure(name) {
    signs.assign(2 * n, -1);
    std::fill(signs.begin(), signs.begin() + static_cast<std::ptrdiff_t>(n), 1);
    std::vector<std::size_t> column_degrees(n);
    for (std::size_t col = 0; col < n; ++col) {
      rows.push_back(static_cast<Index>(col));
      starts.push_back(static_cast<Index>(rows.size()));
    }
    for (std::size_t row = 0; row < n; ++row) {
      std::vector<Index> columns;
      if (name == "ring_band") {
        for (int j = 0; j < width; ++j)
          columns.push_back(static_cast<Index>((row + n + static_cast<std::size_t>(j) - width / 2) % n));
      } else {
        const std::size_t begin = row / 16 * 16, remaining = 16 - static_cast<std::size_t>(width - 1);
        for (int j = 0; j < width - 1; ++j) columns.push_back(static_cast<Index>(begin + j));
        columns.push_back(static_cast<Index>(begin + width - 1 + row % remaining));
      }
      std::sort(columns.begin(), columns.end());
      require(columns.size() == static_cast<std::size_t>(width) &&
          std::adjacent_find(columns.begin(), columns.end()) == columns.end(), "合成G的行项重复");
      for (Index col : columns) { rows.push_back(col); ++column_degrees[static_cast<std::size_t>(col)]; }
      rows.push_back(static_cast<Index>(n + row));
      starts.push_back(static_cast<Index>(rows.size()));
    }
    for (auto count : column_degrees) contributions += count * (count + 1) / 2;
    require(contributions < 8000000 && rows.size() == n * static_cast<std::size_t>(width + 2),
        "合成结构超过冻结贡献预算或非零项数不符");
  }
  double entry(std::size_t col, std::size_t source, int epoch) const {
    const auto row = static_cast<std::size_t>(rows[source]);
    if (row == col) {
      const auto coordinate = col < n ? col : col - n;
      const auto shift = static_cast<std::size_t>(epoch) * (col < n ? 3 : 5);
      const double value = 1.0 + static_cast<double>((coordinate + shift) % 16) / 16.0;
      return col < n ? value : -value;
    }
    const auto dual = col - n;
    const double magnitude = static_cast<double>((dual + 3 * row) % 7 + 1) / 16.0;
    return ((dual + row) & 1) ? -magnitude : magnitude;
  }
};

struct Record {
  bool passed = true;
  bool transfer_ledger_complete = true;
  std::string error;
  Index elapsed_ns = 0, setup_ns = 0, warmup_ns = 0, destruction_ns = 0, total_ns = 0;
  std::uint64_t factor_calls = 0, solve_calls = 0, total_factor_calls = 0, total_solve_calls = 0;
  std::uint64_t budget_calls = 0, constructors = 0, copied_entries = 0, exchanged_entries = 0;
  std::uint64_t warmup_copied_entries = 0, warmup_exchanged_entries = 0, factor_nnz_sum = 0;
  Digest result_digest, input_digest, budget_digest, factor_digest;
  long double max_forward_error = 0, max_residual_error = 0;
};
struct Budget {
  Record& record;
  std::uint64_t phase = 0, epoch = 0;
  void operator()() {
    ++record.budget_calls;
    record.budget_digest.add(phase); record.budget_digest.add(epoch);
    record.budget_digest.add(record.budget_calls);
  }
};
struct State {
  std::vector<double> values;
  std::unique_ptr<Factor> factor;
  bool owned = false;
  std::uint64_t owned_hash = 0;
  State(const Layout& layout, Budget& budget) : values(layout.rows.size()) {
    ++budget.record.constructors;
    factor = std::make_unique<Factor>(2 * layout.n, layout.starts, layout.rows, layout.signs,
        std::ref(budget), false, false, 0, solver::detail::NormalCholeskyKernel::cholmod_supernodal_aat);
  }
};

template<class Value>
void multiply(const Layout& layout, std::span<const double> point, std::vector<long double>& image, Value value) {
  image.assign(2 * layout.n, 0);
  for (std::size_t col = 0; col < 2 * layout.n; ++col) {
    for (Index p = layout.starts[col]; p < layout.starts[col + 1]; ++p) {
      const auto source = static_cast<std::size_t>(p), row = static_cast<std::size_t>(layout.rows[source]);
      const long double coefficient = value(col, source);
      image[row] += coefficient * point[col];
      if (row != col) image[col] += coefficient * point[row];
    }
  }
}

void run_epoch(const Layout& layout, State& state, bool exchange, int epoch, bool measured,
    Record& record, Budget& budget) {
  budget.epoch = static_cast<std::uint64_t>(epoch);
  for (std::size_t col = 0; col < 2 * layout.n; ++col)
    for (Index p = layout.starts[col]; p < layout.starts[col + 1]; ++p)
      state.values[static_cast<std::size_t>(p)] = layout.entry(col, static_cast<std::size_t>(p), epoch);
  std::array<std::vector<double>, 3> points, right_sides;
  std::vector<long double> image;
  // 三个右端都在交出当前矩阵之前生成，之后不再读取Caller中的旧数值作验收。
  for (std::size_t rhs = 0; rhs < points.size(); ++rhs) {
    points[rhs].resize(2 * layout.n); right_sides[rhs].resize(2 * layout.n);
    for (std::size_t i = 0; i < 2 * layout.n; ++i) {
      const double value = static_cast<double>((i + 2 * rhs) % 9 + 1) / 16.0;
      points[rhs][i] = (i + rhs) & 1 ? -value : value;
    }
    multiply(layout, points[rhs], image, [&](std::size_t, std::size_t p) { return state.values[p]; });
    for (std::size_t i = 0; i < image.size(); ++i) right_sides[rhs][i] = static_cast<double>(image[i]);
  }
  const auto before_pointer = state.values.data();
  const auto current_hash = contents(state.values);
  budget.phase = 1;
  ++record.total_factor_calls;
  if (measured) ++record.factor_calls;
  auto& copied = measured ? record.copied_entries : record.warmup_copied_entries;
  auto& exchanged = measured ? record.exchanged_entries : record.warmup_exchanged_entries;
  bool success = false;
  try {
    success = exchange ? state.factor->factor_exchange(state.values, std::ref(budget)) :
        state.factor->factor(state.values, std::ref(budget));
  } catch (...) {
    record.transfer_ledger_complete = false;
    if (state.values.data() != before_pointer) exchanged += static_cast<std::uint64_t>(layout.rows.size());
    throw;
  }
  const bool transferred = state.values.data() != before_pointer;
  const auto returned_hash = contents(state.values);
  if (transferred) exchanged += static_cast<std::uint64_t>(layout.rows.size());
  else if (success) copied += static_cast<std::uint64_t>(layout.rows.size());
  if (!success) record.transfer_ledger_complete = false;
  require(success && state.values.size() == layout.rows.size(), "真实Normal分解失败或交换后Caller尺寸改变");
  require(transferred == (exchange && state.owned), "实际指针轮换与首copy、暖交换分支不符");
  require(returned_hash == (transferred ? state.owned_hash : current_hash), "Caller内容摘要不符合实际所有权交接");
  state.owned = true; state.owned_hash = current_hash;
  if (measured) {
    record.input_digest.add(current_hash);
    record.factor_nnz_sum += static_cast<std::uint64_t>(state.factor->factor_nonzeros());
    record.factor_digest.add(static_cast<std::uint64_t>(state.factor->factor_nonzeros()));
    record.factor_digest.add(static_cast<std::uint64_t>(state.factor->factor_dimension()));
    record.factor_digest.add(static_cast<std::uint64_t>(state.factor->regularized_pivots()));
  }
  for (std::size_t rhs = 0; rhs < points.size(); ++rhs) {
    auto answer = right_sides[rhs];
    budget.phase = 2 + rhs;
    ++record.total_solve_calls;
    if (measured) ++record.solve_calls;
    require(state.factor->solve(answer, std::ref(budget)), "真实Normal回代失败");
    multiply(layout, answer, image, [&](std::size_t col, std::size_t p) { return layout.entry(col, p, epoch); });
    for (std::size_t i = 0; i < answer.size(); ++i) {
      require(std::isfinite(answer[i]) && std::isfinite(image[i]), "解或独立真实乘积非有限");
      const long double forward = std::abs(static_cast<long double>(answer[i]) - points[rhs][i]) /
          std::max(1.0L, std::abs(static_cast<long double>(points[rhs][i])));
      const long double residual = std::abs(image[i] - right_sides[rhs][i]) /
          std::max(1.0L, std::abs(static_cast<long double>(right_sides[rhs][i])));
      record.max_forward_error = std::max(record.max_forward_error, forward);
      record.max_residual_error = std::max(record.max_residual_error, residual);
      require(forward <= tolerance && residual <= tolerance, "已知解或独立真实残差超过1e-11");
      if (measured) record.result_digest.add(answer[i]);
    }
  }
  if (measured) {
    const auto stats = state.factor->refinement_statistics();
    record.factor_digest.add(static_cast<std::uint64_t>(stats.residual_products));
    record.factor_digest.add(static_cast<std::uint64_t>(stats.refinement_steps));
    record.factor_digest.add(static_cast<std::uint64_t>(stats.refinement_acceptances));
  }
}

Record sample(const Layout& layout, bool exchange, bool warm) {
  Record record;
  Budget budget{record};
  std::optional<State> state;
  const auto total_begin = Clock::now();
  Clock::time_point begin{};
  bool timing = false;
  const auto setup = [&](int epoch) {
    const auto start = Clock::now();
    budget.phase = 0; budget.epoch = static_cast<std::uint64_t>(epoch);
    state.emplace(layout, budget);
    record.setup_ns += ns(start, Clock::now());
  };
  const auto destroy = [&] {
    if (!state) return;
    const auto start = Clock::now();
    state.reset();
    record.destruction_ns += ns(start, Clock::now());
  };
  try {
    if (warm) {
      setup(0);
      const auto preheat_begin = Clock::now();
      for (int epoch = 0; epoch < warmup; ++epoch) run_epoch(layout, *state, exchange, epoch, false, record, budget);
      record.warmup_ns = ns(preheat_begin, Clock::now());
    }
    begin = Clock::now(); timing = true;
    for (int epoch = warmup; epoch < warmup + epochs; ++epoch) {
      if (!warm) setup(epoch);
      run_epoch(layout, *state, exchange, epoch, true, record, budget);
      if (!warm) destroy();
    }
    record.elapsed_ns = ns(begin, Clock::now()); timing = false;
    destroy();
  } catch (const std::exception& error) {
    record.passed = false; record.transfer_ledger_complete = false; record.error = error.what();
    destroy();
    if (timing) record.elapsed_ns = ns(begin, Clock::now());
  } catch (...) {
    record.passed = false; record.transfer_ledger_complete = false; record.error = "未知C++异常";
    destroy();
    if (timing) record.elapsed_ns = ns(begin, Clock::now());
  }
  record.total_ns = ns(total_begin, Clock::now());
  return record;
}

std::string quoted(const std::string& value) {
  std::ostringstream stream;
  stream << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') stream << '\\' << static_cast<char>(c);
    else if (c < 32) stream << "\\u" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(c);
    else stream << static_cast<char>(c);
  }
  stream << '"';
  return stream.str();
}
void emit(const Record& r, int sequence, int cell, int block, int position, const std::string& structure,
    std::size_t size, int width, bool warm, bool exchange, std::uint64_t contributions) {
  std::cout << "{\"type\":\"sample\",\"schema\":1,\"sequence\":" << sequence << ",\"cell\":" << cell
      << ",\"structure\":" << quoted(structure) << ",\"size\":" << size << ",\"width\":" << width
      << ",\"lifecycle\":\"" << (warm ? "warm" : "cold") << "\",\"block\":" << block
      << ",\"mode\":\"" << (exchange ? "exchange" : "copy") << "\",\"position\":" << position
      << ",\"passed\":" << (r.passed ? "true" : "false") << ",\"error\":" << quoted(r.error)
      << ",\"transfer_ledger_complete\":" << (r.transfer_ledger_complete ? "true" : "false")
      << ",\"epochs\":8,\"warmup_epochs\":" << (warm ? 3 : 0) << ",\"rhs_per_factor\":3"
      << ",\"dimension\":" << 2 * size << ",\"input_nonzeros\":" << size * static_cast<std::size_t>(width + 2)
      << ",\"gram_contributions\":" << contributions;
  const auto integer = [&](const char* name, auto value) { std::cout << ",\"" << name << "\":" << value; };
  integer("elapsed_ns", r.elapsed_ns); integer("setup_ns", r.setup_ns); integer("warmup_ns", r.warmup_ns);
  integer("destruction_ns", r.destruction_ns); integer("total_ns", r.total_ns);
  integer("factor_calls", r.factor_calls); integer("solve_calls", r.solve_calls);
  integer("total_factor_calls", r.total_factor_calls); integer("total_solve_calls", r.total_solve_calls);
  integer("budget_calls", r.budget_calls); integer("constructors", r.constructors);
  integer("copied_entries", r.copied_entries); integer("exchanged_entries", r.exchanged_entries);
  integer("warmup_copied_entries", r.warmup_copied_entries); integer("warmup_exchanged_entries", r.warmup_exchanged_entries);
  integer("copy_payload_bytes", 8 * r.copied_entries); integer("copy_logical_read_write_bytes", 16 * r.copied_entries);
  integer("factor_nnz_sum", r.factor_nnz_sum);
  std::cout << ",\"result_digest\":" << quoted(r.result_digest.text()) << ",\"input_digest\":" << quoted(r.input_digest.text())
      << ",\"budget_digest\":" << quoted(r.budget_digest.text()) << ",\"factor_digest\":" << quoted(r.factor_digest.text())
      << ",\"max_forward_error\":" << std::setprecision(17);
  const auto number = [](long double value) {
    const double narrowed = static_cast<double>(value);
    if (std::isfinite(narrowed)) std::cout << narrowed;
    else std::cout << "null";
  };
  number(r.max_forward_error);
  std::cout << ",\"max_residual_error\":";
  number(r.max_residual_error);
  std::cout << "}\n";
}
}  // 匿名命名空间

int main() {
  solver::detail::ipm_coupled_state::Log context;
  context.enabled = true;
  solver::detail::ipm_coupled_state::Scope scope(&context);
  std::cout << "{\"type\":\"metadata\",\"schema\":1,\"round\":127,\"cells\":12,\"samples\":432,"
      "\"kernel\":\"cholmod_supernodal_aat\",\"coupled_context_enabled\":true,\"threads\":1,"
      "\"digest\":\"xor_multiply_u64\",\"digest_is_proof\":false,\"public_lp_runs\":0}\n";
  int sequence = 0, failures = 0, cell = 0;
  for (const std::string structure : {"ring_band", "block_arrow_16"})
    for (std::size_t size : {64u, 4096u, 65536u}) for (int width : {3, 9}) {
      std::optional<Layout> layout;
      std::string layout_error;
      try { layout.emplace(size, width, structure); }
      catch (const std::exception& error) { layout_error = error.what(); }
      for (int lifecycle = 0; lifecycle < 2; ++lifecycle) for (int block = 0; block < blocks; ++block)
        for (int position = 0; position < 2; ++position) {
          const bool exchange = ((cell + lifecycle + block) & 1) ? position == 0 : position == 1;
          Record result;
          if (layout) result = sample(*layout, exchange, lifecycle == 1);
          else { result.passed = false; result.transfer_ledger_complete = false; result.error = layout_error; }
          failures += !result.passed;
          emit(result, sequence++, cell, block, position, structure, size, width, lifecycle == 1,
              exchange, layout ? layout->contributions : 0);
        }
      ++cell;
    }
  std::cout << "{\"type\":\"summary\",\"schema\":1,\"samples\":" << sequence << ",\"failures\":" << failures
      << ",\"passed\":" << (failures == 0 && sequence == 432 ? "true" : "false") << "}\n";
  return failures == 0 && sequence == 432 ? 0 : 1;
}
