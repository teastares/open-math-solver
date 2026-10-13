#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace solver::detail {

// 同一算子的物理方向 U 与像 C；只做有界投影和增广，不计算 harmonic Ritz 向量。
class IpmKrylovHistory {
 public:
  using BudgetCheck = std::function<void()>;

  explicit IpmKrylovHistory(std::size_t capacity = 0) : capacity_(capacity) {
    if (capacity > 8) throw std::invalid_argument("Krylov 历史维数必须为 0 到 8");
  }

  std::size_t capacity() const { return capacity_; }
  std::size_t size() const { return images_.size(); }
  void clear() { directions_.clear(); images_.clear(); }
  const std::vector<double>& direction(std::size_t index) const { return directions_.at(index); }
  const std::vector<double>& image(std::size_t index) const { return images_.at(index); }

  // 只有新向量全部通过检查后才提交；满池按已正交化向量的插入顺序淘汰最早项。
  bool add(std::span<const double> direction, std::span<const double> image,
           const BudgetCheck& budget_check = {}) {
    if (budget_check) budget_check();
    if (capacity_ == 0) return false;
    if (direction.size() != image.size() || (!images_.empty() && image.size() != images_[0].size())) {
      throw std::invalid_argument("Krylov 历史入池维数不一致");
    }
    long double original_norm = 0;
    if (!length(image, original_norm) || !(original_norm > 0) || !valid(direction)) return false;
    std::vector<double> next_direction(direction.begin(), direction.end());
    std::vector<double> next_image(image.begin(), image.end());
    if (!subtract_space(next_direction, next_image, budget_check)) return false;
    long double norm = 0;
    if (!length(next_image, norm) ||
        !(norm > std::sqrt(static_cast<long double>(std::numeric_limits<double>::epsilon())) * original_norm)) {
      return false;
    }
    for (std::size_t i = 0; i < next_image.size(); ++i) {
      if (!representable(static_cast<long double>(next_image[i]) / norm, next_image[i]) ||
          !representable(static_cast<long double>(next_direction[i]) / norm, next_direction[i])) return false;
    }
    if (budget_check) budget_check();
    // 其余列本来互相正交，删除最早列无需改变剩余 U/C 的配对关系。
    if (images_.size() == capacity_) {
      directions_.erase(directions_.begin());
      images_.erase(images_.begin());
    }
    directions_.push_back(std::move(next_direction));
    images_.push_back(std::move(next_image));
    return true;
  }

  // 返回投影修正及预测残差；调用方必须另算真实算子残差，不能凭此直接验收。
  bool project(std::span<const double> residual, std::vector<double>& correction,
               std::vector<double>& projected_residual, const BudgetCheck& budget_check = {}) const {
    if (budget_check) budget_check();
    if (images_.empty()) return false;
    if (residual.size() != images_[0].size()) throw std::invalid_argument("Krylov 历史投影维数不一致");
    if (!valid(residual)) return false;
    std::vector<double> next_correction(residual.size(), 0);
    std::vector<double> next_residual(residual.begin(), residual.end());
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t col = 0; col < images_.size(); ++col) {
        if (budget_check) budget_check();
        long double coefficient = 0;
        if (!inner(images_[col], next_residual, coefficient)) return false;
        for (std::size_t i = 0; i < residual.size(); ++i) {
          if (!representable(static_cast<long double>(next_residual[i]) - coefficient * images_[col][i], next_residual[i]) ||
              !representable(static_cast<long double>(next_correction[i]) + coefficient * directions_[col][i], next_correction[i])) {
            return false;
          }
        }
      }
    }
    if (budget_check) budget_check();
    correction = std::move(next_correction);
    projected_residual = std::move(next_residual);
    return true;
  }

  // 对物理 p 与 Kp 同步去除 U/C 分量；数值失败不更改输入，预算异常原样传播。
  bool orthogonalize(std::vector<double>& direction, std::vector<double>& image,
                     const BudgetCheck& budget_check = {}) const {
    if (budget_check) budget_check();
    if (direction.size() != image.size() || (!images_.empty() && image.size() != images_[0].size())) {
      throw std::invalid_argument("Krylov 历史正交化维数不一致");
    }
    if (!valid(direction) || !valid(image)) return false;
    auto next_direction = direction;
    auto next_image = image;
    if (!subtract_space(next_direction, next_image, budget_check)) return false;
    if (budget_check) budget_check();
    direction = std::move(next_direction);
    image = std::move(next_image);
    return true;
  }

 private:
  static bool representable(long double value, double& output) {
    const double converted = static_cast<double>(value);
    if (!std::isfinite(value) || !std::isfinite(converted) || (value != 0 && converted == 0)) return false;
    output = converted;
    return true;
  }

  static bool valid(std::span<const double> values) {
    return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
  }

  static bool length(std::span<const double> values, long double& norm) {
    double scale = 0;
    for (double value : values) {
      if (!std::isfinite(value)) return false;
      scale = std::max(scale, std::abs(value));
    }
    if (scale == 0) { norm = 0; return true; }
    long double sum = 0;
    for (double value : values) {
      const long double relative = static_cast<long double>(value) / scale;
      sum += relative * relative;
    }
    norm = static_cast<long double>(scale) * std::sqrt(sum);
    return std::isfinite(norm);
  }

  static bool inner(std::span<const double> left, std::span<const double> right, long double& sum) {
    sum = 0;
    for (std::size_t i = 0; i < left.size(); ++i) {
      sum += static_cast<long double>(left[i]) * right[i];
      if (!std::isfinite(sum)) return false;
    }
    return true;
  }

  bool subtract_space(std::vector<double>& direction, std::vector<double>& image,
                      const BudgetCheck& budget_check) const {
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t col = 0; col < images_.size(); ++col) {
        if (budget_check) budget_check();
        long double coefficient = 0;
        if (!inner(images_[col], image, coefficient)) return false;
        for (std::size_t i = 0; i < image.size(); ++i) {
          if (!representable(static_cast<long double>(image[i]) - coefficient * images_[col][i], image[i]) ||
              !representable(static_cast<long double>(direction[i]) - coefficient * directions_[col][i], direction[i])) return false;
        }
      }
    }
    return true;
  }

  std::size_t capacity_;
  std::vector<std::vector<double>> directions_, images_;
};

}  // 命名空间 solver::detail
