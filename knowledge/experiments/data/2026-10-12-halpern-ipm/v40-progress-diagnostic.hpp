#pragma once

// 本文件在冻结源码的 Point 定义后、solver 匿名命名空间内引入。
// 所有输入只读；诊断异常只关闭诊断，不改变求解器控制流或结果。
struct V40ProgressState {
  bool enabled = false;
  std::int64_t iteration = 0, event_id = 0, csc_scans = 0, products = 0;
  const char* phase = "initial";
  Clock::time_point solve_start{};
  double completed_seconds = 0;
  std::vector<long double> primal, dual, predicted_primal, predicted_dual;
};

V40ProgressState v40_state;

void v40_configure() noexcept {
  const char* value = std::getenv("SOLVER_V40_PROGRESS");
  v40_state.enabled = value && value[0] == '1' && value[1] == '\0';
}

template <class T> long double v40_norm(const std::vector<T>& values) {
  long double norm = 0;
  for (const auto value : values) {
    if (!std::isfinite(value)) return std::numeric_limits<long double>::quiet_NaN();
    norm = std::max(norm, std::abs(static_cast<long double>(value)));
  }
  return norm;
}

template <class A, class B>
long double v40_difference(const std::vector<A>& first, const std::vector<B>& second) {
  if (first.size() != second.size()) return std::numeric_limits<long double>::quiet_NaN();
  long double norm = 0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    const long double value = static_cast<long double>(first[i]) - second[i];
    if (!std::isfinite(value)) return std::numeric_limits<long double>::quiet_NaN();
    norm = std::max(norm, std::abs(value));
  }
  return norm;
}

struct V40Event {
  std::ostringstream stream;
  bool finite_fields = true;
  Clock::time_point started = Clock::now();
  explicit V40Event(const char* event) {
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<long double>::max_digits10);
    stream << "{\"schema\":1,\"event\":\"" << event << "\",\"phase\":\""
           << v40_state.phase << "\",\"event_id\":" << ++v40_state.event_id
           << ",\"iteration\":" << v40_state.iteration;
    number("elapsed_seconds", elapsed(v40_state.solve_start));
    number("diagnostic_seconds_completed_before_event", v40_state.completed_seconds);
  }
  void boolean(const char* key, bool value) {
    stream << ",\"" << key << "\":" << (value ? "true" : "false");
  }
  void number(const char* key, long double value) {
    // JSON 消费端通常使用双精度；不能把长双精度有限数输出成消费端的无穷或零。
    const long double magnitude = std::abs(value);
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
  void work(const SolveResult& result) {
    optional("completed_iterations", result.ipm_iterations);
    optional("factorizations", result.ipm_factorizations);
    optional("refinement_steps", result.ipm_refinement_steps);
    optional("direction_retries", result.ipm_direction_retries);
    optional("direction_retry_successes", result.ipm_direction_retry_successes);
    optional("primal_regularization", result.ipm_primal_regularization);
    optional("dual_regularization", result.ipm_dual_regularization);
    optional("schur_residual_products", result.ipm_schur_residual_products);
    optional("schur_refinement_steps", result.ipm_schur_refinement_steps);
  }
  void emit() {
    number("diagnostic_csc_scans", v40_state.csc_scans);
    number("diagnostic_vector_products", v40_state.products);
    number("event_compute_seconds", elapsed(started));
    boolean("finite_fields", finite_fields);
    stream << '}';
    const std::string line = "V40_PROGRESS " + stream.str() + "\n";
    std::fwrite(line.data(), 1, line.size(), stderr);
  }
};

template <class Function> void v40_observe(Function&& function) noexcept {
  if (!v40_state.enabled) return;
  const auto start = Clock::now();
  try {
    function();
  } catch (...) {
    v40_state.enabled = false;
    std::fputs("V40_PROGRESS {\"schema\":1,\"event\":\"diagnostic_error\",\"finite_fields\":false}\n", stderr);
  }
  v40_state.completed_seconds += elapsed(start);
}

// 一个稀疏列扫描同时形成 Gx 与 Gᵀz，使用独立的长双精度累加。
void v40_products(const ConeModel& model, const Point& point,
    std::vector<long double>& lower, std::vector<long double>& upper,
    const std::vector<double>* diagonal = nullptr, std::vector<long double>* kkt_lower = nullptr) {
  lower.assign(model.rhs.size(), 0);
  upper.assign(model.costs.size(), 0);
  if (kkt_lower) {
    kkt_lower->resize(model.rhs.size());
    for (std::size_t row = 0; row < model.rhs.size(); ++row)
      (*kkt_lower)[row] = -static_cast<long double>((*diagonal)[row]) * point.z[row];
  }
  for (std::size_t col = 0; col < model.costs.size(); ++col) {
    long double column_product = 0;
    for (Index k = model.starts[col]; k < model.starts[col + 1]; ++k) {
      const auto pos = static_cast<std::size_t>(k);
      const auto row = static_cast<std::size_t>(model.rows[pos]);
      const long double value = model.values[pos];
      const long double contribution = value * point.x[col];
      lower[row] += contribution;
      if (kkt_lower) (*kkt_lower)[row] += contribution;
      column_product += value * point.z[row];
    }
    upper[col] = column_product;
  }
  ++v40_state.csc_scans;
  // KKT 下块另有一条累加链，虽然复用乘法，也计入观测工作账本。
  v40_state.products += kkt_lower ? 3 : 2;
}

void v40_residuals(const ConeModel& model, const Point& point,
    std::vector<long double>& primal, std::vector<long double>& dual) {
  v40_products(model, point, primal, dual);
  for (std::size_t row = 0; row < primal.size(); ++row)
    primal[row] += static_cast<long double>(point.s[row]) - model.rhs[row];
  for (std::size_t col = 0; col < dual.size(); ++col) dual[col] += model.costs[col];
}

void v40_point_fields(V40Event& event, const ConeModel& model, const Point& point,
    const std::vector<long double>& primal, const std::vector<long double>& dual) {
  event.number("x_inf", v40_norm(point.x));
  event.number("s_inf", v40_norm(point.s));
  event.number("z_inf", v40_norm(point.z));
  event.number("rp_ld_inf", v40_norm(primal));
  event.number("rd_ld_inf", v40_norm(dual));
  long double raw = 0, dual_term = 0, primal_term = 0, comp = 0;
  for (std::size_t col = 0; col < point.x.size(); ++col) {
    raw += static_cast<long double>(model.costs[col]) * point.x[col];
    dual_term += static_cast<long double>(point.x[col]) * dual[col];
  }
  for (std::size_t row = 0; row < point.s.size(); ++row) {
    raw += static_cast<long double>(model.rhs[row]) * point.z[row];
    comp += static_cast<long double>(point.s[row]) * point.z[row];
    primal_term -= static_cast<long double>(point.z[row]) * primal[row];
  }
  event.number("working_gap_raw", raw);
  event.number("working_gap_dual_term", dual_term);
  event.number("working_gap_primal_term", primal_term);
  event.number("working_gap_complementarity", comp);
  event.number("working_gap_identity_error", raw - (dual_term + primal_term + comp));
  event.number("objective_scale", model.objective_scale);
  event.number("mu_ld", model.barrier_rows ? comp / model.barrier_rows : 0);
}

void v40_run_start(const SolveOptions& options, Clock::time_point start) noexcept {
  if (!v40_state.enabled) return;
  v40_state.iteration = v40_state.event_id = v40_state.csc_scans = v40_state.products = 0;
  v40_state.phase = "initial";
  v40_state.solve_start = start;
  v40_state.completed_seconds = 0;
  v40_observe([&] {
    V40Event event("run_start");
    event.stream << ",\"linear_accuracy\":\"" << options.ipm.linear_accuracy << '\"';
    event.number("long_double_digits", std::numeric_limits<long double>::digits);
    event.number("time_limit", options.time_limit);
    event.emit();
  });
}

void v40_begin(const ConeModel& model, const Point& point,
    const std::vector<double>& primal_rhs, const std::vector<double>& dual_rhs,
    double mu, std::int64_t iteration, const SolveOptions&, const SolveResult& result) noexcept {
  v40_observe([&] {
    v40_state.iteration = iteration;
    v40_state.phase = "begin";
    V40Event event("begin");
    v40_residuals(model, point, v40_state.primal, v40_state.dual);
    v40_point_fields(event, model, point, v40_state.primal, v40_state.dual);
    event.number("rp_algorithm_inf", v40_norm(primal_rhs));
    event.number("rd_algorithm_inf", v40_norm(dual_rhs));
    event.number("rp_algorithm_ld_difference", v40_difference(primal_rhs, v40_state.primal));
    event.number("rd_algorithm_ld_difference", v40_difference(dual_rhs, v40_state.dual));
    event.number("mu_algorithm", mu);
    event.work(result);
    event.emit();
  });
}

void v40_before_step(const ConeModel& model, const Point& point, const Point& delta,
    const std::vector<double>& primal_rhs, const std::vector<double>& dual_rhs,
    const std::vector<double>& diagonal, const std::vector<double>& h,
    double alpha_primal, double alpha_dual, double sigma, double mu, std::int64_t iteration,
    int regularization_level, const SolveOptions& options, const SolveResult& result) noexcept {
  v40_observe([&] {
    v40_state.iteration = iteration;
    v40_state.phase = "before_step";
    V40Event event("before_step");
    std::vector<long double> gx, gtz, kkt_lower;
    v40_products(model, delta, gx, gtz, &diagonal, &kkt_lower);
    std::vector<long double> ep(gx.size()), ep_true(gx.size()), ed(gtz.size()), ed_true(gtz.size());
    std::vector<long double> ec(gx.size()), lower_rhs(gx.size()), kkt_upper(gtz.size());
    v40_state.predicted_primal.resize(gx.size());
    v40_state.predicted_dual.resize(gtz.size());
    for (std::size_t row = 0; row < gx.size(); ++row) {
      ep[row] = static_cast<long double>(primal_rhs[row]) + gx[row] + delta.s[row];
      ep_true[row] = v40_state.primal[row] + gx[row] + delta.s[row];
      if (model.inequality[row]) ec[row] = static_cast<long double>(point.z[row]) * delta.s[row] +
          static_cast<long double>(point.s[row]) * delta.z[row] - h[row];
      // 与 direction 的有限双精度右端转换一致，不调用会抛异常的生产辅助函数。
      lower_rhs[row] = static_cast<double>(-static_cast<long double>(primal_rhs[row]) -
          (model.inequality[row] ? static_cast<long double>(h[row]) / point.z[row] : 0));
      kkt_lower[row] = lower_rhs[row] - kkt_lower[row];
      v40_state.predicted_primal[row] = (1 - static_cast<long double>(alpha_primal)) *
          v40_state.primal[row] + static_cast<long double>(alpha_primal) * ep_true[row];
    }
    for (std::size_t col = 0; col < gtz.size(); ++col) {
      ed[col] = static_cast<long double>(dual_rhs[col]) + gtz[col];
      ed_true[col] = v40_state.dual[col] + gtz[col];
      kkt_upper[col] = -ed[col];
      v40_state.predicted_dual[col] = (1 - static_cast<long double>(alpha_dual)) *
          v40_state.dual[col] + static_cast<long double>(alpha_dual) * ed_true[col];
    }
    event.number("ep_algorithm_inf", v40_norm(ep));
    event.number("ed_algorithm_inf", v40_norm(ed));
    event.number("ep_ld_inf", v40_norm(ep_true));
    event.number("ed_ld_inf", v40_norm(ed_true));
    event.number("h_complementarity_defect_inf", v40_norm(ec));
    event.number("h_inf", v40_norm(h));
    event.number("diagonal_h_inf", v40_norm(diagonal));
    event.number("dx_inf", v40_norm(delta.x));
    event.number("ds_inf", v40_norm(delta.s));
    event.number("dz_inf", v40_norm(delta.z));
    event.number("alpha_primal", alpha_primal);
    event.number("alpha_dual", alpha_dual);
    event.number("sigma", sigma);
    event.number("mu_algorithm", mu);
    event.number("regularization_level", regularization_level);
    event.number("predicted_rp_ld_inf", v40_norm(v40_state.predicted_primal));
    event.number("predicted_rd_ld_inf", v40_norm(v40_state.predicted_dual));
    const double upper_norm = static_cast<double>(v40_norm(dual_rhs));
    const double lower_norm = static_cast<double>(v40_norm(lower_rhs));
    const double global_tolerance = 1e-12 + 1e-10 * std::max(upper_norm, lower_norm);
    event.number("kkt_rhs_upper_inf", upper_norm);
    event.number("kkt_rhs_lower_inf", lower_norm);
    event.number("kkt_upper_ld_inf", v40_norm(kkt_upper));
    event.number("kkt_lower_ld_inf", v40_norm(kkt_lower));
    event.number("kkt_global_tolerance", global_tolerance);
    event.number("kkt_strict_upper_tolerance", 1e-12 + 1e-10 * upper_norm);
    event.number("kkt_strict_lower_tolerance", 1e-12 + 1e-10 * lower_norm);
    event.boolean("strict_block_tolerances_active", options.ipm.linear_accuracy == "blockwise");
    event.boolean("h_matches_direction", options.ipm.central_corrections == 0);
    event.work(result);
    event.emit();
  });
}

void v40_after_step(const ConeModel& model, const Point& point, std::int64_t iteration,
    const SolveOptions&, const SolveResult& result) noexcept {
  v40_observe([&] {
    v40_state.iteration = iteration;
    v40_state.phase = "after_step";
    V40Event event("after_step");
    std::vector<long double> primal, dual;
    v40_residuals(model, point, primal, dual);
    v40_point_fields(event, model, point, primal, dual);
    event.number("rp_update_prediction_drift", v40_difference(primal, v40_state.predicted_primal));
    event.number("rd_update_prediction_drift", v40_difference(dual, v40_state.predicted_dual));
    event.work(result);
    event.emit();
  });
}

// 直接读取本次已有的原模型验算，不读取可能保留上一候选的 result.lp_optimality。
void v40_current_assess(const LpVerification& checks, const SolveResult& result) noexcept {
  v40_observe([&] {
    V40Event event("current_assess");
    event.boolean("original_valid_input", checks.primal.valid_input && checks.optimality.valid_input);
    event.boolean("original_primal_feasible", checks.primal.feasible);
    event.boolean("original_dual_feasible", checks.optimality.dual_feasible);
    event.boolean("original_gap_satisfied", checks.optimality.gap_satisfied);
    event.boolean("original_numerically_optimal", checks.optimality.numerically_optimal);
    event.number("original_max_row_violation", checks.primal.max_row_violation);
    event.number("original_max_bound_violation", checks.primal.max_bound_violation);
    event.number("original_max_dual_residual", checks.optimality.max_dual_residual);
    event.optional("original_objective", checks.primal.objective);
    event.optional("original_dual_estimate", checks.optimality.dual_objective_estimate);
    event.optional("original_corrected_dual_bound", checks.optimality.corrected_dual_bound);
    event.optional("original_gap", checks.optimality.objective_gap);
    event.optional("original_relative_gap", checks.optimality.relative_objective_gap);
    event.work(result);
    event.emit();
  });
}
