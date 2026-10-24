// 同一份已舍入上三角 Gram 的独立回放，不包含 LP 迭代或新的正则项。
#include <Eigen/SparseCholesky>
#include <cholmod.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Index = std::int64_t;
using LD = long double;
template<class T> using Matrix = Eigen::SparseMatrix<T, Eigen::ColMajor, Index>;
template<class T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
constexpr std::uint64_t input_limit = 128ULL * 1024 * 1024;

void quoted(std::ostream& out, const std::string& value) {
  out << '"';
  const char* digits = "0123456789abcdef";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
    else if (c < 32) out << "\\u00" << digits[c >> 4] << digits[c & 15];
    else out << static_cast<char>(c);
  }
  out << '"';
}
// LD 指标以十进制字符串保存，不让合法 LD 在 JSON 消费端变成无穷。
void metric(std::ostream& out, LD value) {
  const bool valid = std::isfinite(value);
  out << "{\"valid\":" << valid << ",\"decimal\":";
  if (valid) {
    std::ostringstream text;
    text << std::scientific << std::setprecision(std::numeric_limits<LD>::max_digits10) << value;
    quoted(out, text.str());
  } else out << "null";
  out << '}';
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Input {
  std::uint64_t n = 0, nnz = 0, attempt = 0;
  std::vector<Index> starts, rows;
  std::vector<double> values;
  void validate() const {
    require(n > 0 && n <= static_cast<std::uint64_t>(std::numeric_limits<Index>::max()), "矩阵维数非法");
    require(starts.size() == n + 1 && rows.size() == nnz && values.size() == nnz, "CSC 数组长度不符");
    require(starts.front() == 0 && starts.back() >= 0 && static_cast<std::uint64_t>(starts.back()) == nnz, "CSC 边界不符");
    for (std::uint64_t col = 0; col < n; ++col) {
      const Index begin = starts[col], end = starts[col + 1];
      require(begin >= 0 && end >= begin && static_cast<std::uint64_t>(end) <= nnz, "CSC 指针不单调");
      Index previous = -1;
      bool diagonal = false;
      for (Index entry = begin; entry < end; ++entry) {
        const Index row = rows[entry];
        require(row > previous && row <= static_cast<Index>(col), "CSC 行索引并非有序上三角");
        require(std::isfinite(values[entry]), "输入含非有限数");
        diagonal = diagonal || row == static_cast<Index>(col);
        previous = row;
      }
      require(diagonal, "CSC 缺少显式对角");
    }
  }
};
Input read_input(const std::string& prefix) {
  static_assert(sizeof(double) == 8 && sizeof(Index) == 8);
  require(std::endian::native == std::endian::little, "回放只接受小端平台");
  std::ifstream file(prefix + ".normal.bin", std::ios::binary | std::ios::ate);
  require(static_cast<bool>(file), "不能打开矩阵快照");
  const auto bytes = file.tellg();
  require(bytes >= 40 && bytes <= static_cast<std::streamoff>(input_limit), "快照长度超过合同或不足");
  file.seekg(0);
  file.exceptions(std::ios::badbit | std::ios::failbit);
  char magic[8]; file.read(magic, 8);
  require(std::string(magic, 8) == "V49GRAM1", "快照版本不符");
  Input input;
  for (auto* field : {&input.n, &input.nnz, &input.attempt}) file.read(reinterpret_cast<char*>(field), 8);
  require(input.n <= (input_limit - 40) / 8 && input.nnz <= (input_limit - 40 - 8 * input.n) / 16, "快照尺寸溢出");
  require(static_cast<std::uint64_t>(bytes) == 40 + 8 * input.n + 16 * input.nnz, "快照字节数不符");
  input.starts.resize(input.n + 1); input.rows.resize(input.nnz); input.values.resize(input.nnz);
  file.read(reinterpret_cast<char*>(input.starts.data()), static_cast<std::streamsize>(8 * input.starts.size()));
  file.read(reinterpret_cast<char*>(input.rows.data()), static_cast<std::streamsize>(8 * input.rows.size()));
  file.read(reinterpret_cast<char*>(input.values.data()), static_cast<std::streamsize>(8 * input.values.size()));
  input.validate();
  return input;
}
template<class T> Matrix<T> matrix(const Input& input) {
  Matrix<T> result(input.n, input.n);
  result.resizeNonZeros(input.nnz);
  std::copy(input.starts.begin(), input.starts.end(), result.outerIndexPtr());
  std::copy(input.rows.begin(), input.rows.end(), result.innerIndexPtr());
  for (std::size_t i = 0; i < input.values.size(); ++i) result.valuePtr()[i] = static_cast<T>(input.values[i]);
  return result;
}

struct MatrixMetrics { LD norm = 0, diagonal_min = 0, diagonal_max = 0, gershgorin_lower = 0; };
MatrixMetrics matrix_metrics(const Input& input) {
  MatrixMetrics result;
  std::vector<LD> row_sum(input.n, 0), diagonal(input.n, 0);
  for (std::uint64_t col = 0; col < input.n; ++col) {
    for (Index entry = input.starts[col]; entry < input.starts[col + 1]; ++entry) {
      const auto row = input.rows[entry];
      const LD value = input.values[entry];
      row_sum[row] += std::abs(value);
      if (row != static_cast<Index>(col)) row_sum[col] += std::abs(value);
      else diagonal[col] = value;
    }
  }
  result.diagonal_min = *std::min_element(diagonal.begin(), diagonal.end());
  result.diagonal_max = *std::max_element(diagonal.begin(), diagonal.end());
  result.gershgorin_lower = std::numeric_limits<LD>::infinity();
  for (std::size_t i = 0; i < input.n; ++i) {
    result.norm = std::max(result.norm, row_sum[i]);
    result.gershgorin_lower = std::min(result.gershgorin_lower, diagonal[i] - (row_sum[i] - std::abs(diagonal[i])));
  }
  return result;
}
struct Residual {
  bool finite = false;
  LD norm = std::numeric_limits<LD>::quiet_NaN();
  LD backward = std::numeric_limits<LD>::quiet_NaN();
  LD xnorm = std::numeric_limits<LD>::quiet_NaN();
};
Residual residual(const Input& input, LD matrix_norm, const std::vector<LD>& x, int rhs_kind) {
  Residual result;
  for (LD value : x) if (!std::isfinite(value)) return result;
  std::vector<LD> work(input.n, 0);
  for (std::uint64_t col = 0; col < input.n; ++col) {
    for (Index entry = input.starts[col]; entry < input.starts[col + 1]; ++entry) {
      const auto row = input.rows[entry];
      const LD value = input.values[entry];
      work[row] += value * x[col];
      if (row != static_cast<Index>(col)) work[col] += value * x[row];
    }
  }
  result.norm = result.xnorm = 0;
  for (std::size_t row = 0; row < input.n; ++row) {
    const LD r = (rhs_kind && row % 2 ? -1.L : 1.L) - work[row];
    if (!std::isfinite(r)) return result;
    result.norm = std::max(result.norm, std::abs(r));
    result.xnorm = std::max(result.xnorm, std::abs(x[row]));
  }
  const LD denominator = matrix_norm * result.xnorm + 1;
  if (std::isfinite(denominator) && denominator > 0) result.backward = result.norm / denominator;
  result.finite = std::isfinite(result.norm) && std::isfinite(result.xnorm);
  return result;
}
void emit_residual(std::ostream& out, const Residual& result) {
  out << "{\"finite\":" << result.finite << ",\"residual_inf\":"; metric(out, result.norm);
  out << ",\"backward_error\":"; metric(out, result.backward);
  out << ",\"solution_inf\":"; metric(out, result.xnorm); out << '}';
}
struct Solve {
  int rhs = 0, status = 0;
  bool success = false, cast_representable = false;
  Residual physical, cast_double;
};
struct Backend {
  std::string name, error;
  bool success = false, finite_factor = false, permutation_identity = true;
  int status = 0, factor_return = 0;
  std::uint64_t minor = 0, stored = 0, nsuper = 0;
  std::uint64_t factor_started = 0, solve_started = 0, residual_products = 0;
  LD diagonal_min = std::numeric_limits<LD>::quiet_NaN(), diagonal_max = std::numeric_limits<LD>::quiet_NaN();
  std::vector<Solve> solves;
};
void assess_solution(Backend& backend, Solve& solve, const Input& input, LD norm, const std::vector<LD>& x, bool cast) {
  ++backend.residual_products;
  solve.physical = residual(input, norm, x, solve.rhs);
  solve.success = solve.physical.finite;
  if (!cast || !solve.success) return;
  std::vector<LD> rounded(x.size());
  solve.cast_representable = true;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double value = static_cast<double>(x[i]);
    if (!std::isfinite(value) || (value == 0 && x[i] != 0)) solve.cast_representable = false;
    rounded[i] = value;
  }
  ++backend.residual_products;
  solve.cast_double = residual(input, norm, rounded, solve.rhs);
}

template<class T> class Checked : public Eigen::SimplicialLLT<Matrix<T>, Eigen::Upper, Eigen::NaturalOrdering<Index>> {
 public:
  bool inspect(Backend& backend) const {
    backend.stored = this->m_matrix.nonZeros();
    LD minimum = std::numeric_limits<LD>::infinity(), maximum = 0;
    for (Index col = 0; col < this->m_matrix.cols(); ++col) {
      bool diagonal = false;
      for (typename Matrix<T>::InnerIterator entry(this->m_matrix, col); entry; ++entry) {
        if (!std::isfinite(entry.value())) return false;
        if (entry.row() == col) {
          if (!(entry.value() > 0)) return false;
          diagonal = true;
          minimum = std::min(minimum, static_cast<LD>(entry.value()));
          maximum = std::max(maximum, static_cast<LD>(entry.value()));
        }
      }
      if (!diagonal) return false;
    }
    backend.diagonal_min = minimum; backend.diagonal_max = maximum;
    return true;
  }
};
template<class T> Backend eigen_replay(const Input& input, LD norm, const char* name) {
  Backend result; result.name = name;
  try {
    const auto gram = matrix<T>(input);
    Checked<T> factor;
    factor.analyzePattern(gram);
    ++result.factor_started;
    factor.factorize(gram);
    result.status = static_cast<int>(factor.info());
    result.factor_return = factor.info() == Eigen::Success;
    if (!result.factor_return) return result;
    result.finite_factor = factor.inspect(result);
    result.success = result.finite_factor;
    if (!result.success) return result;
    for (int kind = 0; kind < 2; ++kind) {
      Solve solve; solve.rhs = kind;
      Vector<T> rhs(input.n);
      for (std::size_t i = 0; i < input.n; ++i) rhs[i] = kind && i % 2 ? -1 : 1;
      ++result.solve_started;
      const Vector<T> answer = factor.solve(rhs).eval();
      solve.status = static_cast<int>(factor.info());
      if (factor.info() == Eigen::Success) {
        std::vector<LD> values(answer.size());
        for (Eigen::Index i = 0; i < answer.size(); ++i) values[i] = answer[i];
        assess_solution(result, solve, input, norm, values, std::is_same_v<T, LD>);
      }
      result.solves.push_back(solve);
    }
  } catch (const std::exception& error) { result.error = error.what(); }
  return result;
}

struct Cholmod {
  cholmod_common common{};
  cholmod_factor* factor = nullptr;
  cholmod_dense *answer = nullptr, *work = nullptr, *extra = nullptr;
  Cholmod() {
    require(cholmod_l_start(&common), "CHOLMOD 初始化失败");
    common.print = 0; common.error_handler = nullptr;
    common.supernodal = CHOLMOD_SUPERNODAL;
    common.nmethods = 1; common.method[0].ordering = CHOLMOD_NATURAL;
    common.postorder = 0; common.final_asis = 1; common.useGPU = 0; common.nthreads_max = 1;
    common.dbound = 0; common.sbound = 0; common.quick_return_if_not_posdef = 1;
  }
  ~Cholmod() {
    cholmod_l_free_dense(&answer, &common); cholmod_l_free_dense(&work, &common);
    cholmod_l_free_dense(&extra, &common); cholmod_l_free_factor(&factor, &common);
    cholmod_l_finish(&common);
  }
};
bool inspect_cholmod(const cholmod_factor& factor, Backend& output) {
  if (!factor.is_super || !factor.is_ll || factor.itype != CHOLMOD_LONG || factor.xtype != CHOLMOD_REAL ||
      factor.dtype != CHOLMOD_DOUBLE || !factor.x) return false;
  const auto* super = static_cast<const Index*>(factor.super);
  const auto* pointers = static_cast<const Index*>(factor.pi);
  const auto* offsets = static_cast<const Index*>(factor.px);
  const auto* values = static_cast<const double*>(factor.x);
  LD minimum = std::numeric_limits<LD>::infinity(), maximum = 0;
  for (std::size_t node = 0; node < factor.nsuper; ++node) {
    const Index columns = super[node + 1] - super[node], rows = pointers[node + 1] - pointers[node], offset = offsets[node];
    if (columns <= 0 || rows < columns || offset < 0 || static_cast<std::size_t>(offset) > factor.xsize ||
        static_cast<std::size_t>(columns) > (factor.xsize - offset) / rows) return false;
    for (Index col = 0; col < columns; ++col) {
      const Index begin = offset + col * rows;
      const double diagonal = values[begin + col];
      if (!(diagonal > 0)) return false;
      for (Index row = col; row < rows; ++row) if (!std::isfinite(values[begin + row])) return false;
      minimum = std::min(minimum, static_cast<LD>(diagonal)); maximum = std::max(maximum, static_cast<LD>(diagonal));
    }
  }
  output.diagonal_min = minimum; output.diagonal_max = maximum;
  return true;
}
Backend cholmod_replay(const Input& input, LD norm) {
  Backend result; result.name = "cholmod_supernodal";
  try {
    Cholmod resources;
    cholmod_sparse gram{};
    gram.nrow = gram.ncol = input.n; gram.nzmax = input.nnz;
    gram.p = const_cast<Index*>(input.starts.data()); gram.i = const_cast<Index*>(input.rows.data());
    gram.stype = 1; gram.itype = CHOLMOD_LONG; gram.xtype = CHOLMOD_PATTERN; gram.dtype = CHOLMOD_DOUBLE;
    gram.sorted = gram.packed = 1;
    resources.factor = cholmod_l_analyze(&gram, &resources.common);
    result.status = resources.common.status;
    if (!resources.factor || result.status != CHOLMOD_OK) return result;
    const auto* permutation = static_cast<const Index*>(resources.factor->Perm);
    result.permutation_identity = permutation != nullptr;
    for (std::size_t i = 0; i < input.n && result.permutation_identity; ++i)
      result.permutation_identity = permutation[i] == static_cast<Index>(i);
    if (!result.permutation_identity) return result;
    gram.xtype = CHOLMOD_REAL; gram.x = const_cast<double*>(input.values.data());
    ++result.factor_started;
    result.factor_return = cholmod_l_factorize(&gram, resources.factor, &resources.common);
    result.status = resources.common.status;
    result.minor = resources.factor->minor; result.stored = resources.factor->xsize; result.nsuper = resources.factor->nsuper;
    if (!result.factor_return || result.status != CHOLMOD_OK || result.minor != input.n) return result;
    result.finite_factor = inspect_cholmod(*resources.factor, result);
    result.success = result.finite_factor;
    if (!result.success) return result;
    for (int kind = 0; kind < 2; ++kind) {
      Solve solve; solve.rhs = kind;
      std::vector<double> rhs(input.n);
      for (std::size_t i = 0; i < input.n; ++i) rhs[i] = kind && i % 2 ? -1 : 1;
      cholmod_dense b{};
      b.nrow = b.d = b.nzmax = input.n; b.ncol = 1; b.x = rhs.data(); b.xtype = CHOLMOD_REAL; b.dtype = CHOLMOD_DOUBLE;
      ++result.solve_started;
      const int status = cholmod_l_solve2(CHOLMOD_A, resources.factor, &b, nullptr, &resources.answer, nullptr,
          &resources.work, &resources.extra, &resources.common);
      solve.status = resources.common.status;
      if (status && solve.status == CHOLMOD_OK && resources.answer) {
        const auto* x = static_cast<const double*>(resources.answer->x);
        assess_solution(result, solve, input, norm, std::vector<LD>(x, x + input.n), false);
      }
      result.solves.push_back(solve);
    }
  } catch (const std::exception& error) { result.error = error.what(); }
  return result;
}
std::vector<Backend> replay(const Input& input, LD norm) {
  // 顺序固定，所有方法读取同一份原始 double 值；不存在重装配或重新排序。
  std::vector<Backend> result;
  result.push_back(cholmod_replay(input, norm));
  result.push_back(eigen_replay<double>(input, norm, "eigen_double"));
  result.push_back(eigen_replay<LD>(input, norm, "eigen_long_double"));
  return result;
}
void emit_backend(std::ostream& out, const Backend& result) {
  out << "{\"name\":"; quoted(out, result.name);
  out << ",\"success\":" << result.success << ",\"finite_factor\":" << result.finite_factor
      << ",\"permutation_identity\":" << result.permutation_identity << ",\"status\":" << result.status
      << ",\"factor_return\":" << result.factor_return << ",\"minor\":";
  if (result.name == "cholmod_supernodal") out << result.minor; else out << "null";
  out << ",\"stored_entries\":" << result.stored << ",\"nsuper\":" << result.nsuper
      << ",\"factor_started\":" << result.factor_started << ",\"solve_started\":" << result.solve_started
      << ",\"residual_products\":" << result.residual_products << ",\"factor_diagonal_min\":";
  metric(out, result.diagonal_min); out << ",\"factor_diagonal_max\":"; metric(out, result.diagonal_max);
  out << ",\"error\":"; if (result.error.empty()) out << "null"; else quoted(out, result.error);
  out << ",\"solves\":[";
  for (std::size_t i = 0; i < result.solves.size(); ++i) {
    if (i) out << ',';
    const auto& solve = result.solves[i];
    out << "{\"rhs\":\"" << (solve.rhs ? "alternating" : "ones") << "\",\"success\":" << solve.success
        << ",\"status\":" << solve.status << ",\"physical\":"; emit_residual(out, solve.physical);
    out << ",\"cast_double\":";
    if (result.name == "eigen_long_double") {
      out << "{\"representable\":" << solve.cast_representable << ",\"residual\":";
      emit_residual(out, solve.cast_double); out << '}';
    } else out << "null";
    out << '}';
  }
  out << "]}";
}
void self_test() {
  Input input{2, 3, 0, {0, 1, 3}, {0, 0, 1}, {4, 1, 3}};
  input.validate();
  const auto norms = matrix_metrics(input);
  require(norms.norm == 5 && norms.diagonal_min == 3 && norms.gershgorin_lower == 2, "矩阵指标手算失败");
  for (int rhs = 0; rhs < 2; ++rhs) {
    const std::vector<LD> known = rhs ? std::vector<LD>{4.L / 11, -5.L / 11} : std::vector<LD>{2.L / 11, 3.L / 11};
    require(residual(input, norms.norm, known, rhs).norm < 1e-17L, "独立真残差不符合解析解");
  }
  const auto good = replay(input, norms.norm);
  for (const auto& backend : good) {
    require(backend.success && backend.factor_started == 1 && backend.solve_started == 2 && backend.solves.size() == 2, "正定解析例未通过");
    for (const auto& solve : backend.solves) require(solve.success && solve.physical.norm < 1e-14L, "正定解析例残差过大");
    if (backend.name == "eigen_long_double") {
      require(backend.residual_products == 4, "LD 和 double 转换后的残差未各自执行");
      for (const auto& solve : backend.solves) require(solve.cast_representable && solve.cast_double.finite, "LD 转换自测失败");
    } else require(backend.residual_products == 2, "直接解真实残差未执行");
  }
  Input indefinite = input; indefinite.values = {1, 2, 1};
  for (const auto& backend : replay(indefinite, matrix_metrics(indefinite).norm))
    require(!backend.success && backend.factor_started == 1 && backend.solve_started == 0, "非正定矩阵错误通过");
  int rejected = 0;
  for (int kind = 0; kind < 3; ++kind) {
    Input invalid = input;
    if (kind == 0) invalid.values[0] = std::numeric_limits<double>::quiet_NaN();
    if (kind == 1) invalid.starts[1] = 4;
    if (kind == 2) invalid.rows[1] = 1;
    try { invalid.validate(); } catch (const std::runtime_error&) { ++rejected; }
  }
  require(rejected == 3, "NaN 或畸形 CSC 未拒绝");
  std::cout << "{\"schema\":1,\"passed\":true,\"spd_backends\":3,\"indefinite_backends\":3,\"invalid_inputs_rejected\":3,\"known_rhs\":2,\"backends\":[";
  for (std::size_t i = 0; i < good.size(); ++i) { if (i) std::cout << ','; emit_backend(std::cout, good[i]); }
  std::cout << "]}\n";
}
}  // 匿名命名空间

int main(int argc, char** argv) {
  std::cout << std::boolalpha;
  try {
    require(argc == 2, "用法：factor-replay 输出前缀，或 --self-test");
    if (std::string(argv[1]) == "--self-test") { self_test(); return 0; }
    const auto input = read_input(argv[1]);
    const auto norms = matrix_metrics(input);
    const auto backends = replay(input, norms.norm);
    std::cout << "{\"schema\":1,\"n\":" << input.n << ",\"nnz\":" << input.nnz << ",\"snapshot_attempt\":" << input.attempt
        << ",\"long_double_digits\":" << std::numeric_limits<LD>::digits << ",\"matrix_metrics\":{\"inf_norm\":";
    metric(std::cout, norms.norm); std::cout << ",\"diagonal_min\":"; metric(std::cout, norms.diagonal_min);
    std::cout << ",\"diagonal_max\":"; metric(std::cout, norms.diagonal_max);
    std::cout << ",\"gershgorin_lower\":"; metric(std::cout, norms.gershgorin_lower); std::cout << "},\"backends\":[";
    for (std::size_t i = 0; i < backends.size(); ++i) { if (i) std::cout << ','; emit_backend(std::cout, backends[i]); }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cout << "{\"schema\":1,\"error\":"; quoted(std::cout, error.what()); std::cout << "}\n";
    return 1;
  }
}
