// 仅诊断：直接包含冻结实现，解析后比较 canonicalize，绝不调用 run_ipm。
#include "frozen/src/ipm.cpp"
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
struct Structure {
  std::size_t singleton_inequalities = 0;
  std::size_t singleton_equalities = 0;
  std::vector<bool> inequality_support;
  std::vector<bool> bound_equality;
};

Structure inspect(const solver::ConeModel& model) {
  Structure result;
  result.inequality_support.assign(model.costs.size(), false);
  result.bound_equality.assign(model.costs.size(), false);
  std::vector<std::size_t> row_nonzeros(model.rhs.size(), 0);
  std::vector<std::size_t> row_column(model.rhs.size(), 0);
  for (std::size_t col = 0; col < model.costs.size(); ++col) {
    for (auto k = model.starts[col]; k < model.starts[col + 1]; ++k) {
      const auto row = static_cast<std::size_t>(model.rows[static_cast<std::size_t>(k)]);
      ++row_nonzeros[row];
      row_column[row] = col;
    }
  }
  for (std::size_t row = 0; row < model.rhs.size(); ++row) {
    if (row_nonzeros[row] != 1) continue;
    const auto col = row_column[row];
    if (model.inequality[row]) {
      ++result.singleton_inequalities;
      result.inequality_support[col] = true;
    } else {
      ++result.singleton_equalities;
      if (model.original_row[row] == -1) result.bound_equality[col] = true;
    }
  }
  return result;
}

void array(std::ostream& stream, const std::vector<std::size_t>& values) {
  stream << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) stream << ',';
    stream << values[i];
  }
  stream << ']';
}

void describe(std::ostream& stream, const solver::ConeModel& model, const Structure& shape) {
  stream << "{\"columns\":" << model.costs.size()
         << ",\"rows\":" << model.rhs.size()
         << ",\"nonzeros\":" << model.values.size()
         << ",\"barrier_rows\":" << model.barrier_rows
         << ",\"equality_rows\":" << model.rhs.size() - model.barrier_rows
         << ",\"singleton_inequality_rows\":" << shape.singleton_inequalities
         << ",\"singleton_equality_rows\":" << shape.singleton_equalities
         << ",\"coordinates_with_singleton_inequality_support\":"
         << std::count(shape.inequality_support.begin(), shape.inequality_support.end(), true)
         << ",\"bound_equality_columns\":"
         << std::count(shape.bound_equality.begin(), shape.bound_equality.end(), true) << '}';
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("需要一个模型路径");
    solver::SolveOptions read_options;
    read_options.backend = "ipm";
    read_options.time_limit = 0;
    const auto read = solver::solve_file(std::filesystem::path(argv[1]), read_options);
    if (read.status != "time_limit" || read.ipm_iterations.value_or(-1) != 0 ||
        read.ipm_factorizations.value_or(-1) != 0 || read.has_solution) {
      throw std::runtime_error("零预算解析路径异常：" + read.status + "：" + read.message);
    }
    solver::SolveOptions options;
    options.backend = "ipm";
    options.time_limit = std::numeric_limits<double>::infinity();
    options.ipm.eliminate_fixed_variables = true;
    options.ipm.eliminate_singletons = true;
    options.ipm.equilibrate_kkt = true;
    options.ipm.linear_solver = "normal_augmented";
    options.ipm.presolve_singleton_rows = false;
    const auto off = solver::canonicalize(read.model, options, solver::Clock::now());
    options.ipm.presolve_singleton_rows = true;
    const auto on = solver::canonicalize(read.model, options, solver::Clock::now());
    if (off.costs.size() != on.costs.size() || off.original_columns != on.original_columns ||
        off.fixed_values != on.fixed_values) {
      throw std::runtime_error("归并开关改变了原有固定变量消除映射");
    }
    const auto before = inspect(off), after = inspect(on);
    std::vector<std::size_t> lost, gained, new_fixed, originally_fixed;
    for (std::size_t col = 0; col < on.costs.size(); ++col) {
      const auto original_col = on.original_columns.empty() ? col : on.original_columns[col];
      if (before.inequality_support[col] && !after.inequality_support[col]) lost.push_back(original_col);
      if (!before.inequality_support[col] && after.inequality_support[col]) gained.push_back(original_col);
      if (after.bound_equality[col] && !before.bound_equality[col]) {
        if (std::isfinite(read.model.column_lower[original_col]) &&
            read.model.column_lower[original_col] == read.model.column_upper[original_col]) {
          originally_fixed.push_back(original_col);
        } else new_fixed.push_back(original_col);
      }
    }
    std::ostringstream stream;
    stream << "{\"diagnostic_only\":true,\"ipm_version\":" << std::quoted(solver::ipm_version())
           << ",\"parser_backend\":\"solve_file/ipm/time0\",\"iterations\":0,\"factorizations\":0"
           << ",\"original_columns\":" << read.model.costs.size()
           << ",\"original_rows\":" << read.model.row_lower.size()
           << ",\"eliminated_input_fixed_variables\":" << read.model.costs.size() - off.costs.size()
           << ",\"presolved_singleton_rows\":" << on.presolved_singleton_rows
           << ",\"off\":";
    describe(stream, off, before);
    stream << ",\"on\":";
    describe(stream, on, after);
    stream << ",\"new_fixed_bound_columns\":" << new_fixed.size()
           << ",\"new_fixed_original_column_indices\":";
    array(stream, new_fixed);
    stream << ",\"new_bound_equalities_on_originally_fixed_columns\":";
    array(stream, originally_fixed);
    stream << ",\"lost_singleton_inequality_support_columns\":" << lost.size()
           << ",\"lost_support_original_column_indices\":";
    array(stream, lost);
    stream << ",\"gained_singleton_inequality_support_columns\":" << gained.size()
           << ",\"gained_support_original_column_indices\":";
    array(stream, gained);
    stream << '}';
    std::cout << stream.str() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
