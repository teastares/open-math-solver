# 现代 LP 的 Accelerate 复测

实测基于 `559afa4`、Apple M5 Pro、macOS 26.5.2 和 Rosetta x86_64。CLI 来自同一组 Release 对象，重链 Apple Accelerate，使用 HiGHS 1.15.1。六个模型来自[已锁定的开发集](../../benchmarks/modern-lp-smoke.lock.json)。各后端单线程、种子 0，原模型通过原始可行性、对偶可行性和目标间隙检查，求解加验算须在数值预算内。每个模型与配置仅运行一次，部分进程并行。耗时和覆盖率对应这组输入与本机配置。

| 方法 | 15 秒内通过 | 60 秒内通过 | 300 秒内通过 |
| --- | ---: | ---: | ---: |
| 自适应 PDLP | 1/6 | 3/6 | 4/6 |
| 齐次内点法，启用 HiGHS 预处理和 IPX 交叉 | 0/6 | 1/6 | 2/6 |
| HiGHS 自动模式 | 未运行 | 0/6 | 2/6 |

- 组合内点法的 ex10、rmine15 触发预测填充上限 16,777,216 项，woodlands09 触发超节点填充或工作区上限。三个模型均在内点迭代前退出。PDLP 在 60 秒内通过 ex10、woodlands09。对 60 秒未通过的三例，补测使用 600 秒与 100 万步上限，rmine15 新增通过，约 594.7 秒墙钟、80,850 步，低于默认 20 万步。irish-electricity 完成 322,045 步，原始可行性、对偶可行性和间隙均未达标。
- 60 秒档的 qap15、woodlands09，PDLP 原模型验算占求解加验算耗时的 69.2% 和 68.6%。300 秒档的 supportcase10，组合内点法耗时 181.1 秒、峰值驻留 941 MiB，PDLP 为 261.0 秒、144 MiB。可据此测试中间候选检查频率，并比较分解资源受限时切换 PDLP 的收益。
- rmine15 在 15、60、300 秒档保留初始可行点，目标 0、相对间隙 1。额外诊断副本在固定 1,000 步下与正式版除计时外的输出一致。120 秒诊断运行的 44 个非初始候选均未达到原始可行性，均值候选的最大行违反量从第 1,000 步的 17.24 降到第 21,146 步的 1.12。600 秒正式运行通过终验的目标为 −5,042.48296158。
- HiGHS 在 irish-electricity 上返回内部 Optimal，包装层的原模型检查报告最大对偶残差 9.87176，未计入通过。该对偶偏差的来源还未确定。

[52 次正式运行](data/modern-lp-accelerate/results.csv) · [内点法阶段计时](data/modern-lp-accelerate/ipm-stages.csv) · [5 个模型的 10 份 PDLP 最优输出独立重算](data/modern-lp-accelerate/independent-check.csv) · [rmine15 诊断副本候选轨迹](data/modern-lp-accelerate/rmine15-candidates.csv)

## 复现

使用 Python 3.11 或更新版本。先按[开发与运行说明](../development.md)使用锁定依赖在 macOS 上构建 x86_64 Release 版本，使用 Ninja，并在配置时指定 `-DCMAKE_OSX_ARCHITECTURES=x86_64`，OpenBLAS 与主程序均使用 x86_64。重链脚本读取 Ninja 的既有链接命令，将静态 OpenBLAS 替换为系统 Accelerate，用于本实验。脚本需要已完成构建的目录。

```sh
python3 scripts/fetch_modern_lp.py
python3 scripts/relink_macos_accelerate.py --build build/release --output build/release/solver-accelerate
python3 scripts/benchmark_modern_accelerate.py pdlp 15 60 300 --jobs 6 \
  --solver build/release/solver-accelerate --models benchmarks/data/modern-lp-smoke --output build/modern-pdlp
```

同一脚本支持 `ipm 15 60 300 --jobs 2`、`highs 60 --jobs 2`、`highs 300 --jobs 4`，每组使用新的输出目录。PDLP 长时限组使用 `pdlp 600 --jobs 3 --cases irish-electricity rmine15 supportcase10 --iteration-limit 1000000`。rmine15 的 Halpern 对照使用 `pdlp 300 --cases rmine15 --pdlp-mode halpern`。脚本核对锁定模型的字节数与 SHA256，保存可执行文件哈希、压缩原始 JSON 和逐次 CSV。原实测 CLI 的 SHA256 为 `0620012bc02126f050f1f39746b9dd3a9133993899dda5fbf1286005a020a1fc`。
