# 单轮开发探针说明

[归档索引](readme.md) · [实验正文](../../2026-10-12-历史信息搜索与CPU回归.md)

两轮均按 `benchmarks/development.txt` 的八模型顺序串行执行，每模型先 `adaptive` 再 `subspace`，各一次，不作交替重复性能结论。每次求解预算 5 秒，接受步数上限 200000，检查间隔 50，缩放 5 轮，历史容量 8，其余使用当时默认选项。

编译命令（在项目根目录执行，第二轮将输出文件名改为 `.tools/history_probe_v2.exe`）：

```powershell
& .tools/llvm-mingw-20261006-ucrt-x86_64/bin/clang++.exe -std=c++20 -O3 -static -Iinclude -Ibuild/release/_deps/highs-src/extern -Ibuild/release/_deps/highs-src -Ibuild/release/_deps/highs-src/highs -Ibuild/release/_deps/highs-build -Ibuild/release/_deps/highs-build/extern src/model.cpp src/verification.cpp src/lp_optimality.cpp src/pdlp.cpp src/highs_backend.cpp .tools/history_probe.cpp build/release/bin/libhighs.a build/release/bin/libhighs_extras.a -o .tools/history_probe.exe
& .tools/history_probe.exe adaptive <模型绝对路径>
& .tools/history_probe.exe subspace <模型绝对路径>
```

没有显式传入 `-DNDEBUG`，没有改动或重新编译 HiGHS 库。探针源码见 [history_probe.cpp](history_probe.cpp)。复现历史版本时需将对应源码快照恢复到独立目录，不能直接用最终源码冒充首轮算法。

| 轮次 | 自主源码身份 | 探针二进制 SHA256 | 输出 |
| --- | --- | --- | --- |
| 首轮 | [实际编译源码](pdlp-subspace-v1-built.cpp)，`46029259c520df263339956e6894a3a96e4cc916a5f7001e836de3dda759f51c` | `53141d451d8dc44b76aa9b276c949c0802ee5b8b639350615b5c475b2fe7e305` | [工具输出逐项整理](history-exploration-v1.json) |
| 第二轮 | [按需分配修复前源码包](before-lazy-source.zip)中的 `src/pdlp.cpp`，`57ea512234f4ba5b381e9931c6f9a95791bb7b849f65cf22e400334b2e9a2f86` | `b24be9d83eb0cf96c07fe44d33ba5c418d4e8a2686c5e86ac3a538fb5d68d7fb` | [原始文本](history-exploration-v2.txt) |

[首轮后补预算检查的源码](pdlp-subspace-v1.cpp)多了一处每组合列的时间预算检查，保留以说明真实编辑过程，不能把它当作首轮实际编译身份。

每行十列依次是：模型路径、模式、状态、接受步数、内部总时间、求解时间、验算时间、搜索尝试数、接受数、内核算子次数。`total_seconds` 来自 `solve_file()`，包含模型解析，不含外部进程启动和终端输出；`solve_seconds` 不含解析及独立验算。数值输出精度为 10 位有效数字，可选整数缺失时输出 `-1`。首轮 JSON 由工具日志整理，第二轮文本用 `Tee-Object` 保存。
