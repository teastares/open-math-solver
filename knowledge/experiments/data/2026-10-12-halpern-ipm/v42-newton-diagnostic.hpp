#pragma once

// 仅在冻结源码的 Point 定义后引入；标准库头由独立 driver 提前引入。
// 本文件只观测既有输入，诊断自身异常只关闭观测，不替换生产异常或方向。
struct V42Work {
  std::int64_t j_started = 0, j_completed = 0, csc_started = 0, csc_completed = 0;
  std::int64_t nnz_visits = 0, metric_elements = 0, copied_elements = 0;
  std::int64_t cached_images = 0, cache_peak_elements = 0, reconstruction_terms = 0;
};

struct V42ScaledNorm {
  long double scale = 0, squares = 1;
  bool valid = true;
  void add(long double value) {
    value = std::abs(value);
    if (!std::isfinite(value)) { valid = false; return; }
    if (value == 0) return;
    if (scale < value) {
      const long double ratio = scale / value;
      squares = 1 + squares * ratio * ratio;
      scale = value;
    } else {
      const long double ratio = value / scale;
      squares += ratio * ratio;
    }
  }
  long double value() const {
    return valid ? scale * std::sqrt(squares) : std::numeric_limits<long double>::quiet_NaN();
  }
};

struct V42Metrics {
  long double weighted_l2 = 0, scaled_inf = 0;
  std::array<long double, 3> block_scaled_inf{}, block_inf{};
};

std::size_t v42_block(const detail::IpmFullNewtonLayout& layout, std::size_t i) {
  return i < layout.columns ? 0 : i < layout.columns + layout.rows ? 1 : 2;
}

template <class Value>
V42Metrics v42_metric_values(const detail::IpmFullNewtonLayout& layout,
    const std::array<double, 3>& tolerance, Value&& value, V42Work* work = nullptr) {
  V42Metrics result;
  V42ScaledNorm norm;
  bool valid = true;
  for (std::size_t i = 0; i < layout.dimension(); ++i) {
    if (work) ++work->metric_elements;
    const auto block = v42_block(layout, i);
    const long double raw = std::abs(value(i));
    const long double scaled = raw / tolerance[block];
    if (!std::isfinite(raw) || !std::isfinite(scaled) || !(tolerance[block] > 0)) valid = false;
    result.block_inf[block] = std::max(result.block_inf[block], raw);
    result.block_scaled_inf[block] = std::max(result.block_scaled_inf[block], scaled);
    result.scaled_inf = std::max(result.scaled_inf, scaled);
    norm.add(scaled);
  }
  result.weighted_l2 = norm.value();
  if (!valid) {
    result.weighted_l2 = result.scaled_inf = std::numeric_limits<long double>::quiet_NaN();
    result.block_inf.fill(result.scaled_inf);
    result.block_scaled_inf.fill(result.scaled_inf);
  }
  return result;
}

// 自测可直接调用；空互补块的范数为零，不增加人为分母下限。
V42Metrics v42_metrics(const detail::IpmFullNewtonLayout& layout,
    std::span<const long double> residual, const std::array<double, 3>& tolerance,
    V42Work* work = nullptr) {
  if (residual.size() != layout.dimension()) throw std::invalid_argument("诊断残差维数不一致");
  return v42_metric_values(layout, tolerance, [&](std::size_t i) { return residual[i]; }, work);
}

V42Metrics v42_residual_difference(const detail::IpmFullNewtonLayout& layout,
    std::span<const long double> first, std::span<const long double> second,
    const std::array<double, 3>& tolerance, V42Work* work = nullptr) {
  if (first.size() != layout.dimension() || second.size() != layout.dimension())
    throw std::invalid_argument("诊断残差差维数不一致");
  return v42_metric_values(layout, tolerance,
      [&](std::size_t i) { return first[i] - second[i]; }, work);
}

// 独立长双精度输入算子，每次只扫描一遍 CSC，同时累加两个稀疏乘积。
std::vector<long double> v42_product(const detail::IpmFullNewtonLayout& layout,
    std::span<const Index> starts, std::span<const Index> rows, std::span<const double> values,
    std::span<const double> slack, std::span<const double> dual,
    std::span<const long double> direction, V42Work* work = nullptr) {
  if (direction.size() != layout.dimension() || starts.size() != layout.columns + 1 ||
      rows.size() != values.size() || slack.size() != layout.rows || dual.size() != layout.rows)
    throw std::invalid_argument("诊断算子维数不一致");
  if (work) ++work->j_started;
  std::vector<long double> result(layout.dimension(), 0);
  if (work) ++work->csc_started;
  for (std::size_t col = 0; col < layout.columns; ++col) {
    long double upper = 0;
    for (Index k = starts[col]; k < starts[col + 1]; ++k) {
      if (work) ++work->nnz_visits;
      const auto pos = static_cast<std::size_t>(k);
      const auto row = static_cast<std::size_t>(rows[pos]);
      const long double value = values[pos];
      upper += value * direction[layout.columns + row];
      result[layout.columns + row] += value * direction[col];
    }
    result[col] = upper;
  }
  if (work) ++work->csc_completed;
  for (std::size_t i = 0; i < layout.inequality_rows.size(); ++i) {
    const auto row = layout.inequality_rows[i];
    const auto index = layout.columns + layout.rows + i;
    result[layout.columns + row] += direction[index];
    result[index] = static_cast<long double>(dual[row]) * direction[index] +
        static_cast<long double>(slack[row]) * direction[layout.columns + row];
  }
  if (work) ++work->j_completed;
  return result;
}

struct V42Reconstruction {
  std::vector<long double> ld_sum;
  std::vector<double> once_cast;
  bool ld_sum_finite = true, once_cast_finite = true;
  std::int64_t once_cast_underflows = 0;
};

V42Reconstruction v42_reconstruct(std::span<const double> base,
    const std::vector<std::vector<double>>& directions, std::span<const double> coefficients,
    V42Work* work = nullptr) {
  if (coefficients.size() > directions.size()) throw std::invalid_argument("诊断系数维数不一致");
  for (std::size_t j = 0; j < coefficients.size(); ++j)
    if (directions[j].size() != base.size()) throw std::invalid_argument("诊断基方向维数不一致");
  V42Reconstruction result;
  result.ld_sum.assign(base.begin(), base.end());
  if (work) work->copied_elements += static_cast<std::int64_t>(base.size());
  result.once_cast.resize(base.size());
  for (std::size_t i = 0; i < base.size(); ++i) {
    for (std::size_t j = 0; j < coefficients.size(); ++j) {
      if (work) ++work->reconstruction_terms;
      result.ld_sum[i] += static_cast<long double>(coefficients[j]) * directions[j][i];
    }
    const long double value = result.ld_sum[i];
    const bool finite_value = std::isfinite(value);
    const bool convertible = finite_value && std::abs(value) <= std::numeric_limits<double>::max();
    result.ld_sum_finite = result.ld_sum_finite && finite_value;
    result.once_cast_finite = result.once_cast_finite && convertible;
    result.once_cast[i] = convertible ? static_cast<double>(value) : std::numeric_limits<double>::quiet_NaN();
    if (convertible && value != 0 && result.once_cast[i] == 0) ++result.once_cast_underflows;
  }
  return result;
}

struct V42State {
  bool enabled = false, active = false, terminal = false, candidate_pending = false;
  std::int64_t event_id = 0, direction_id = 0, iteration = 0, candidate_id = 0;
  std::int64_t preconditioner_started = 0, preconditioner_completed = 0;
  std::int64_t product_started = 0, product_completed = 0;
  std::int64_t direction_refinement_steps = 0, direction_krylov_steps = 0;
  std::int64_t candidates_started = 0, candidates_completed = 0;
  int restart = -1, column = -1, ir_iteration = 0;
  const char* phase = "unknown";
  const char* stage = "direction";
  const ConeModel* model = nullptr;
  const Point* point = nullptr;
  const detail::IpmFullNewtonLayout* layout = nullptr;
  const SolveResult* result = nullptr;
  std::span<const double> rhs;
  std::array<double, 3> tolerance{};
  std::vector<double> base;
  std::vector<long double> base_residual;
  std::vector<std::vector<long double>> images;
  const std::vector<std::vector<double>>* directions = nullptr;
  std::span<const double> coefficients;
  double estimate = 0, remaining = 0;
  bool trigger_window = false;
  V42Metrics last_metrics;
  V42Work work;
  Clock::time_point solve_start{};
  double diagnostic_seconds = 0;
};

V42State v42_state;

void v42_configure() noexcept {
  const char* value = std::getenv("SOLVER_V42_NEWTON");
  v42_state.enabled = value && value[0] == '1' && value[1] == '\0';
}

struct V42Event {
  std::ostringstream stream;
  bool finite_fields = true;
  Clock::time_point started = Clock::now();
  static void quote(std::ostream& output, const char* value) {
    output << '"';
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
      if (*p == '"' || *p == '\\') output << '\\' << static_cast<char>(*p);
      else if (*p < 32) output << "\\u00" << "0123456789abcdef"[*p >> 4] << "0123456789abcdef"[*p & 15];
      else output << static_cast<char>(*p);
    }
    output << '"';
  }
  void string(const char* key, const char* value) {
    stream << ",\"" << key << "\":";
    quote(stream, value);
  }
  void integer(const char* key, std::int64_t value) { stream << ",\"" << key << "\":" << value; }
  void boolean(const char* key, bool value) { stream << ",\"" << key << "\":" << (value ? "true" : "false"); }
  void number(const char* key, long double value) {
    const auto magnitude = std::abs(value);
    const bool valid = std::isfinite(value) && magnitude <= std::numeric_limits<double>::max() &&
        (value == 0 || magnitude >= std::numeric_limits<double>::denorm_min());
    stream << ",\"" << key << "\":";
    if (valid) stream << value; else stream << "null";
    stream << ",\"" << key << "_valid\":" << (valid ? "true" : "false");
    finite_fields = finite_fields && valid;
  }
  template <class T> void optional(const char* key, const std::optional<T>& value) {
    if (value) number(key, *value);
    else stream << ",\"" << key << "\":null,\"" << key << "_valid\":false";
  }
  explicit V42Event(const char* event) {
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<long double>::max_digits10);
    stream << "{\"schema\":1,\"event\":";
    quote(stream, event);
    integer("event_id", ++v42_state.event_id);
    integer("direction_id", v42_state.direction_id);
    integer("outer_iteration", v42_state.iteration);
    string("phase", v42_state.phase);
    string("stage", v42_state.stage);
    integer("restart", v42_state.restart);
    integer("column", v42_state.column);
    integer("ir_iteration", v42_state.ir_iteration);
    integer("candidate_id", v42_state.candidate_id);
    boolean("candidate_pending", v42_state.candidate_pending);
    number("elapsed_seconds", elapsed(v42_state.solve_start));
    number("diagnostic_seconds_completed_before_event", v42_state.diagnostic_seconds);
  }
  void metrics(const char* prefix, const V42Metrics& value) {
    const std::string p(prefix);
    number((p + "_weighted_l2").c_str(), value.weighted_l2);
    number((p + "_scaled_inf").c_str(), value.scaled_inf);
    constexpr const char* names[] = {"dual", "primal", "complementarity"};
    for (std::size_t i = 0; i < 3; ++i) {
      number((p + "_" + names[i] + "_scaled_inf").c_str(), value.block_scaled_inf[i]);
      number((p + "_" + names[i] + "_inf").c_str(), value.block_inf[i]);
    }
  }
  void emit() {
    integer("observed_preconditioner_started", v42_state.preconditioner_started);
    integer("observed_preconditioner_completed", v42_state.preconditioner_completed);
    integer("observed_products_started", v42_state.product_started);
    integer("observed_products_completed", v42_state.product_completed);
    integer("direction_refinement_steps", v42_state.direction_refinement_steps);
    integer("direction_krylov_steps", v42_state.direction_krylov_steps);
    integer("candidates_started", v42_state.candidates_started);
    integer("candidates_completed", v42_state.candidates_completed);
    if (v42_state.result) {
      const auto& result = *v42_state.result;
      optional("production_preconditioner_started", result.ipm_full_newton_preconditioner_solves);
      optional("production_products_started", result.ipm_full_newton_products);
      optional("reported_refinement_steps", result.ipm_refinement_steps);
      optional("reported_krylov_steps", result.ipm_weighted_krylov_steps);
      optional("factorizations", result.ipm_factorizations);
      optional("primal_regularization", result.ipm_primal_regularization);
      optional("dual_regularization", result.ipm_dual_regularization);
    }
    const auto& work = v42_state.work;
    integer("diagnostic_j_started", work.j_started);
    integer("diagnostic_j_completed", work.j_completed);
    integer("diagnostic_csc_started", work.csc_started);
    integer("diagnostic_csc_completed", work.csc_completed);
    integer("diagnostic_nnz_visits", work.nnz_visits);
    integer("diagnostic_metric_elements", work.metric_elements);
    integer("diagnostic_copied_elements", work.copied_elements);
    integer("diagnostic_cached_images", work.cached_images);
    integer("diagnostic_cache_peak_elements", work.cache_peak_elements);
    integer("diagnostic_reconstruction_terms", work.reconstruction_terms);
    number("event_compute_seconds", elapsed(started));
    boolean("finite_fields", finite_fields);
    stream << '}';
    const std::string line = "V42_NEWTON " + stream.str() + "\n";
    std::fwrite(line.data(), 1, line.size(), stderr);
  }
};

template <class Function> void v42_observe(Function&& function) noexcept {
  if (!v42_state.enabled) return;
  const auto started = Clock::now();
  try { function(); }
  catch (...) {
    v42_state.enabled = false;
    // 错误路径使用固定缓冲区，保留已开始和已完成的乘积账本。
    char buffer[768];
    const int size = std::snprintf(buffer, sizeof(buffer),
        "V42_NEWTON {\"schema\":1,\"event\":\"diagnostic_error\",\"event_id\":%lld,"
        "\"direction_id\":%lld,\"outer_iteration\":%lld,\"candidate_pending\":%s,"
        "\"diagnostic_j_started\":%lld,\"diagnostic_j_completed\":%lld,"
        "\"diagnostic_csc_started\":%lld,\"diagnostic_csc_completed\":%lld,\"finite_fields\":false}\n",
        static_cast<long long>(++v42_state.event_id), static_cast<long long>(v42_state.direction_id),
        static_cast<long long>(v42_state.iteration), v42_state.candidate_pending ? "true" : "false",
        static_cast<long long>(v42_state.work.j_started), static_cast<long long>(v42_state.work.j_completed),
        static_cast<long long>(v42_state.work.csc_started), static_cast<long long>(v42_state.work.csc_completed));
    if (size > 0 && static_cast<std::size_t>(size) < sizeof(buffer)) std::fwrite(buffer, 1, size, stderr);
  }
  v42_state.diagnostic_seconds += elapsed(started);
}

void v42_run_start(Clock::time_point start) noexcept {
  if (!v42_state.enabled) return;
  v42_state.solve_start = start;
}

void v42_phase(const char* phase, std::int64_t iteration) noexcept {
  if (!v42_state.enabled) return;
  v42_state.phase = phase;
  v42_state.iteration = iteration;
}

void v42_work(const detail::IpmFullNewtonWork& work) noexcept {
  if (!v42_state.enabled) return;
  v42_state.direction_refinement_steps = work.refinement_steps;
  v42_state.direction_krylov_steps = work.krylov_steps;
}

struct V42DirectionScope {
  int exceptions = 0;
  explicit V42DirectionScope(const ConeModel& model, const Point& point,
      const detail::IpmFullNewtonLayout& layout, std::span<const double> rhs,
      const SolveResult& result) noexcept {
    if (!v42_state.enabled) return;
    exceptions = std::uncaught_exceptions();
    v42_observe([&] {
      ++v42_state.direction_id;
      v42_state.active = true;
      v42_state.terminal = v42_state.candidate_pending = false;
      v42_state.candidate_id = 0;
      v42_state.preconditioner_started = v42_state.preconditioner_completed = 0;
      v42_state.product_started = v42_state.product_completed = 0;
      v42_state.direction_refinement_steps = v42_state.direction_krylov_steps = 0;
      v42_state.candidates_started = v42_state.candidates_completed = 0;
      v42_state.restart = v42_state.column = -1;
      v42_state.ir_iteration = 0;
      v42_state.stage = "direction";
      v42_state.model = &model; v42_state.point = &point; v42_state.layout = &layout;
      v42_state.rhs = rhs; v42_state.result = &result;
      v42_state.base.clear(); v42_state.base_residual.clear(); v42_state.images.clear();
      v42_state.directions = nullptr; v42_state.coefficients = {};
    });
  }
  ~V42DirectionScope() noexcept {
    if (!v42_state.enabled || !v42_state.active || v42_state.terminal) return;
    v42_observe([&] {
      const bool failed = std::uncaught_exceptions() > exceptions;
      V42Event event(failed ? "direction_failure" : "direction_return");
      event.string("failure_kind", failed ? "uncaught_exception" : "none");
      if (!failed) event.metrics("true", v42_state.last_metrics);
      event.emit();
      v42_state.terminal = true;
      v42_state.active = false;
    });
  }
};

void v42_direction_failure(const detail::IpmFullNewtonWork& work) noexcept {
  if (!v42_state.enabled) return;
  v42_work(work);
  v42_observe([&] {
    V42Event event("direction_failure");
    try {
      const auto exception = std::current_exception();
      if (exception) std::rethrow_exception(exception);
      event.string("failure_kind", "unknown");
    } catch (const BudgetExpired&) {
      event.string("failure_kind", "time_limit");
    } catch (const NumericalFailure& failure) {
      event.string("failure_kind", "numerical_failure"); event.string("message", failure.what());
    } catch (const std::exception& failure) {
      event.string("failure_kind", "other_exception"); event.string("message", failure.what());
    } catch (...) { event.string("failure_kind", "unknown_exception"); }
    event.emit();
    v42_state.terminal = true;
    v42_state.active = false;
  });
}

void v42_helper_begin(const std::array<double, 3>& tolerance) noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  v42_observe([&] {
    v42_state.tolerance = tolerance;
    v42_state.stage = "initial";
    V42Event event("direction_begin");
    event.integer("columns", static_cast<std::int64_t>(v42_state.layout->columns));
    event.integer("rows", static_cast<std::int64_t>(v42_state.layout->rows));
    event.integer("inequalities", static_cast<std::int64_t>(v42_state.layout->inequality_rows.size()));
    event.integer("nonzeros", static_cast<std::int64_t>(v42_state.model->values.size()));
    event.number("dual_tolerance", tolerance[0]);
    event.number("primal_tolerance", tolerance[1]);
    event.number("complementarity_tolerance", tolerance[2]);
    event.emit();
  });
}

void v42_preconditioner_begin() noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  ++v42_state.preconditioner_started;
}
void v42_preconditioner_end() noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  ++v42_state.preconditioner_completed;
}
void v42_product_begin() noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  ++v42_state.product_started;
}
void v42_product_end() noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  ++v42_state.product_completed;
}

void v42_iteration_check(int iteration, const detail::IpmFullNewtonWork& work) noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  v42_work(work);
  v42_state.stage = iteration == 0 ? "initial" : "ir";
  v42_state.ir_iteration = iteration;
}

void v42_ir_begin(int iteration) noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  v42_state.stage = "ir"; v42_state.ir_iteration = iteration;
}

void v42_restart(int restart, const std::vector<double>& base,
    const std::vector<long double>& residual, const detail::IpmFullNewtonWork& work) noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  v42_work(work);
  v42_observe([&] {
    v42_state.stage = "krylov"; v42_state.restart = restart; v42_state.column = -1;
    v42_state.base = base; v42_state.base_residual = residual; v42_state.images.clear();
    v42_state.work.copied_elements += static_cast<std::int64_t>(base.size() + residual.size());
    V42Event event("restart");
    event.metrics("true", v42_metrics(*v42_state.layout, residual, v42_state.tolerance, &v42_state.work));
    event.emit();
  });
}

void v42_arnoldi_begin(std::size_t column) noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  v42_state.stage = "arnoldi"; v42_state.column = static_cast<int>(column);
}

void v42_arnoldi_image(std::size_t column, const std::vector<long double>& image,
    const detail::IpmFullNewtonWork& work) noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  v42_work(work);
  v42_observe([&] {
    if (column != v42_state.images.size()) throw std::invalid_argument("诊断像缓存列序不一致");
    v42_state.images.push_back(image);
    ++v42_state.work.cached_images;
    v42_state.work.copied_elements += static_cast<std::int64_t>(image.size());
    v42_state.work.cache_peak_elements = std::max(v42_state.work.cache_peak_elements,
        static_cast<std::int64_t>(v42_state.images.size() * image.size()));
  });
}

void v42_arnoldi_started(const detail::IpmFullNewtonWork& work) noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  v42_work(work);
  v42_observe([&] { V42Event event("arnoldi_start"); event.emit(); });
}

void v42_arnoldi_end(double remaining, double estimate) noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  v42_observe([&] {
    V42Event event("arnoldi");
    event.number("remaining", remaining); event.number("givens_weighted_l2", std::abs(estimate));
    event.emit();
  });
}

void v42_candidate_prepare(const std::vector<std::vector<double>>& directions,
    const std::vector<double>& coefficients, double estimate, double remaining, bool window) noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  v42_state.stage = "candidate"; v42_state.candidate_pending = true;
  ++v42_state.candidate_id; ++v42_state.candidates_started;
  v42_state.directions = &directions; v42_state.coefficients = coefficients;
  v42_state.estimate = std::abs(estimate); v42_state.remaining = remaining;
  v42_state.trigger_window = window;
}

template <class First, class Second>
void v42_direction_difference(V42Event& event, const char* prefix,
    const First& first, const Second& second) {
  V42ScaledNorm norm;
  long double inf = 0;
  bool valid = true;
  for (std::size_t i = 0; i < first.size(); ++i) {
    ++v42_state.work.metric_elements;
    const long double value = static_cast<long double>(first[i]) - second[i];
    if (!std::isfinite(value)) valid = false;
    inf = std::max(inf, std::abs(value)); norm.add(value);
  }
  const std::string p(prefix);
  event.number((p + "_l2").c_str(), norm.value());
  event.number((p + "_inf").c_str(), valid ? inf : std::numeric_limits<long double>::quiet_NaN());
}

void v42_assessed(const std::vector<double>& answer, const std::vector<long double>& residual,
    long double measure) noexcept {
  if (!v42_state.enabled || !v42_state.active) return;
  v42_observe([&] {
    const auto& layout = *v42_state.layout;
    const auto& tolerance = v42_state.tolerance;
    v42_state.last_metrics = v42_metrics(layout, residual, tolerance, &v42_state.work);
    const bool candidate = v42_state.candidate_pending;
    if (candidate) { ++v42_state.candidates_completed; v42_state.candidate_pending = false; }
    V42Event event(candidate ? "candidate" : std::string_view(v42_state.stage) == "initial" ? "initial_check" : "ir_check");
    event.metrics("true", v42_state.last_metrics);
    event.number("production_measure", measure);
    event.boolean("accepted", measure <= 1);
    if (candidate) {
      event.number("givens_weighted_l2", v42_state.estimate);
      event.number("remaining", v42_state.remaining);
      event.boolean("trigger_estimate", v42_state.estimate <= 1);
      event.boolean("trigger_window", v42_state.trigger_window);
      event.boolean("trigger_remaining_zero", v42_state.remaining == 0);
      const auto& coefficients = v42_state.coefficients;
      if (coefficients.size() > v42_state.images.size()) throw std::invalid_argument("诊断像缓存缺列");
      auto image_residual = v42_state.base_residual;
      v42_state.work.copied_elements += static_cast<std::int64_t>(image_residual.size());
      for (std::size_t j = 0; j < coefficients.size(); ++j)
        for (std::size_t i = 0; i < image_residual.size(); ++i) {
          ++v42_state.work.reconstruction_terms;
          image_residual[i] -= static_cast<long double>(coefficients[j]) * v42_state.images[j][i];
        }
      event.metrics("image", v42_metrics(layout, image_residual, tolerance, &v42_state.work));
      event.metrics("image_minus_true", v42_residual_difference(layout, image_residual, residual, tolerance, &v42_state.work));
      const auto reconstruction = v42_reconstruct(v42_state.base, *v42_state.directions,
          coefficients, &v42_state.work);
      event.boolean("ld_sum_finite", reconstruction.ld_sum_finite);
      event.boolean("once_cast_finite", reconstruction.once_cast_finite);
      event.integer("once_cast_underflows", reconstruction.once_cast_underflows);
      v42_direction_difference(event, "current_minus_ld_sum", answer, reconstruction.ld_sum);
      v42_direction_difference(event, "once_cast_minus_ld_sum", reconstruction.once_cast, reconstruction.ld_sum);
      v42_direction_difference(event, "current_minus_once_cast", answer, reconstruction.once_cast);
      const auto& model = *v42_state.model;
      const auto& point = *v42_state.point;
      auto ld_residual = v42_product(layout, model.starts, model.rows, model.values,
          point.s, point.z, reconstruction.ld_sum, &v42_state.work);
      std::vector<long double> once_direction(reconstruction.once_cast.begin(), reconstruction.once_cast.end());
      v42_state.work.copied_elements += static_cast<std::int64_t>(once_direction.size());
      auto once_residual = v42_product(layout, model.starts, model.rows, model.values,
          point.s, point.z, once_direction, &v42_state.work);
      for (std::size_t i = 0; i < layout.dimension(); ++i) {
        ld_residual[i] = static_cast<long double>(v42_state.rhs[i]) - ld_residual[i];
        once_residual[i] = static_cast<long double>(v42_state.rhs[i]) - once_residual[i];
      }
      event.metrics("ld_sum", v42_metrics(layout, ld_residual, tolerance, &v42_state.work));
      event.metrics("once_cast", v42_metrics(layout, once_residual, tolerance, &v42_state.work));
      event.metrics("ld_sum_minus_true", v42_residual_difference(layout, ld_residual, residual, tolerance, &v42_state.work));
      event.metrics("once_cast_minus_true", v42_residual_difference(layout, once_residual, residual, tolerance, &v42_state.work));
      event.metrics("once_cast_minus_ld_sum", v42_residual_difference(layout, once_residual, ld_residual, tolerance, &v42_state.work));
      event.metrics("image_minus_ld_sum", v42_residual_difference(layout, image_residual, ld_residual, tolerance, &v42_state.work));
    }
    event.emit();
  });
}
