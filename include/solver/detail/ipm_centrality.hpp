#pragma once

#include <functional>
#include <span>
#include <vector>

namespace solver::detail {

// 内部数值接口；各行向量长度一致，锥行的当前原始和对偶变量严格为正。
struct IpmCentralityTarget {
  std::vector<double> target;
  double primal_step = 1;
  double dual_step = 1;
};

struct IpmCentralityWeights {
  double primal_weight = 0;
  double dual_weight = 0;
};

IpmCentralityTarget ipm_centrality_target(
    std::span<const double> slack, std::span<const double> dual,
    std::span<const double> slack_direction, std::span<const double> dual_direction,
    const std::vector<bool>& inequality, double target_mu,
    const std::function<void()>& budget_check = {});

IpmCentralityWeights ipm_centrality_weights(
    std::span<const double> slack, std::span<const double> dual,
    std::span<const double> slack_direction, std::span<const double> dual_direction,
    std::span<const double> slack_correction, std::span<const double> dual_correction,
    const std::vector<bool>& inequality, double old_primal_step, double old_dual_step,
    const std::function<void()>& budget_check = {});

}  // 命名空间 solver::detail
