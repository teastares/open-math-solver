# 历史信息搜索实验归档索引

[实验正文](../../2026-10-12-历史信息搜索与CPU回归.md)

| 批次 | 清单 | 汇总或中断说明 | 完整已写入记录 |
| --- | --- | --- | --- |
| 最终开发集内部消融 | [manifest](development-ablation/manifest.json) | [summary](development-ablation/summary.json) | [JSONL gzip](development-ablation/runs.jsonl.gz) |
| 最终开发集 HiGHS 对照 | [manifest](development-highs/manifest.json) | [summary](development-highs/summary.json) | [JSONL gzip](development-highs/runs.jsonl.gz) |
| 最终扩大回归内部消融 | [manifest](regression-ablation/manifest.json) | [summary](regression-ablation/summary.json) | [JSONL gzip](regression-ablation/runs.jsonl.gz) |
| 最终扩大回归 HiGHS 对照 | [manifest](regression-highs/manifest.json) | [summary](regression-highs/summary.json) | [JSONL gzip](regression-highs/runs.jsonl.gz) |
| 按需分配修正前开发消融 | [manifest](before-lazy-development-ablation/manifest.json) | [summary](before-lazy-development-ablation/summary.json) | [JSONL gzip](before-lazy-development-ablation/runs.jsonl.gz) |
| 按需分配修正前开发 HiGHS 对照 | [manifest](before-lazy-development-highs/manifest.json) | [summary](before-lazy-development-highs/summary.json) | [JSONL gzip](before-lazy-development-highs/runs.jsonl.gz) |
| 主动中止的修正前扩大回归，45 条完成记录 | [manifest](before-lazy-partial-regression-ablation/manifest.json) | [中断说明](before-lazy-partial-regression-ablation/interruption.json) | [JSONL gzip](before-lazy-partial-regression-ablation/runs.jsonl.gz) |

| 身份与检查 | 文件 |
| --- | --- |
| 最终身份与源码 | [identity](identity.json)、[逐文件哈希](source-sha256.json)、[源码快照](source.zip) |
| 按需分配修正前身份 | [identity](before-lazy-identity.json)、[逐文件哈希](before-lazy-source-sha256.json)、[源码快照](before-lazy-source.zip) |
| 最终构建与正确性 | [构建日志](build.log)、[CTest 日志](ctest.log) |
| 修正前构建与正确性 | [构建日志](before-lazy-build.log)、[CTest 日志](before-lazy-ctest.log) |
| 环境 | [硬件快照](hardware.json)、[依赖锁](dependencies.lock.json) |
| 两轮开发探索 | [方法、编译参数及输出定义](exploration.md) |
| 缓存复用前后数值轨迹审计 | [adaptive-equivalence.json](adaptive-equivalence.json) |
| 归档完整性、计数与耗时 | [archive-audit.json](archive-audit.json) |

主实验使用最终身份，不把不同版本数据合并。压缩记录是原始 JSONL 的无损 gzip，失败、未完成及原始输出全部保留。被中断批次仅有 45 条已完成记录，没有生成完整集合成绩；中断原因是修正额外缓存分配，不是据其模型结果调算法。
