#include <Highs.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

// 只读诊断：按固定变量代入及单非零不等式消元后的模式计数，不运行 LP 求解。
int main(int argc, char** argv) {
  for (int argument = 1; argument < argc; ++argument) {
    Highs reader;
    reader.setOptionValue("output_flag", false);
    if (reader.readModel(argv[argument]) != HighsStatus::kOk) return 2;
    const auto& lp = reader.getLp();
    const auto& a = lp.a_matrix_;
    std::vector<bool> fixed(lp.num_col_);
    std::vector<std::int64_t> row_degree(lp.num_row_, 0);
    std::vector<std::vector<HighsInt>> columns(lp.num_col_);
    std::int64_t fixed_count = 0;
    for (HighsInt col = 0; col < lp.num_col_; ++col) {
      fixed[col] = lp.col_lower_[col] == lp.col_upper_[col] &&
          lp.col_lower_[col] > -kHighsInf && lp.col_upper_[col] < kHighsInf;
      if (fixed[col]) { ++fixed_count; continue; }
      std::vector<std::pair<HighsInt, double>> entries;
      for (HighsInt k = a.start_[col]; k < a.start_[col + 1]; ++k) entries.emplace_back(a.index_[k], a.value_[k]);
      std::sort(entries.begin(), entries.end());
      for (std::size_t i = 0; i < entries.size();) {
        const auto row = entries[i].first;
        long double sum = 0;
        do { sum += entries[i++].second; } while (i < entries.size() && entries[i].first == row);
        if (sum != 0) { columns[col].push_back(row); ++row_degree[row]; }
      }
    }
    std::vector<std::int64_t> row_copies(lp.num_row_);
    std::int64_t retained = 0, empty_rows = 0;
    for (HighsInt row = 0; row < lp.num_row_; ++row) {
      if (lp.row_lower_[row] == lp.row_upper_[row]) row_copies[row] = 1;
      else if (row_degree[row] != 1) row_copies[row] = (lp.row_lower_[row] > -kHighsInf) + (lp.row_upper_[row] < kHighsInf);
      retained += row_copies[row];
      if (row_degree[row] == 0) empty_rows += row_copies[row];
    }
    std::vector<std::int64_t> degrees;
    std::int64_t contributions = retained, nonzeros = 0;
    for (HighsInt col = 0; col < lp.num_col_; ++col) {
      if (fixed[col]) continue;
      std::int64_t degree = 0;
      for (auto row : columns[col]) degree += row_copies[row];
      contributions += degree * (degree + 1) / 2;
      nonzeros += degree;
      degrees.push_back(degree);
    }
    std::sort(degrees.rbegin(), degrees.rend());
    std::cout << "{\"model\":\"" << lp.model_name_ << "\",\"rows\":" << lp.num_row_
              << ",\"columns\":" << lp.num_col_ << ",\"fixed\":" << fixed_count
              << ",\"retained\":" << retained << ",\"empty_retained\":" << empty_rows
              << ",\"g_nonzeros\":" << nonzeros << ",\"contributions\":" << contributions << ",\"top_degrees\":[";
    for (std::size_t i = 0; i < std::min<std::size_t>(degrees.size(), 10); ++i) std::cout << (i ? "," : "") << degrees[i];
    std::cout << "]}\n";
  }
}
