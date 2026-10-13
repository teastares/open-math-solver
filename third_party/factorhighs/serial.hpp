#pragma once

#include <utility>

// 独立分解版本只执行同步任务，不接触 HiGHS 的线程调度器。
namespace solver::factor_parallel {
class TaskGroup {
 public:
  template <class Function> void spawn(Function&& function) const {
    std::forward<Function>(function)();
  }
  void sync() const {}
  void taskWait() const {}
};

template <class Index, class Function>
void for_each(Index first, Index last, Function&& function, Index) {
  if (first < last) std::forward<Function>(function)(first, last);
}
}  // 命名空间 solver::factor_parallel
