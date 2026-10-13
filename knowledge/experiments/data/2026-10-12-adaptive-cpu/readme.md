# 自适应 CPU 实验归档索引

[实验正文](../../2026-10-12-自适应CPU优化与Netlib验证.md)

| 主题 | 清单 | 汇总 | 完整逐次记录 |
| --- | --- | --- | --- |
| 最终开发集 | [manifest](development/manifest.json) | [summary](development/summary.json) | [JSONL 压缩归档](development/runs.jsonl.gz) |
| 最终扩大验证 | [manifest](validation/manifest.json) | [summary](validation/summary.json) | [JSONL 压缩归档](validation/runs.jsonl.gz) |
| 最终固定步长消融 | [manifest](baseline-control/manifest.json) | [summary](baseline-control/summary.json) | [JSONL 压缩归档](baseline-control/runs.jsonl.gz) |
| 投影兼容修复前开发集 | [manifest](before-dual-projection-development/manifest.json) | [summary](before-dual-projection-development/summary.json) | [JSONL 压缩归档](before-dual-projection-development/runs.jsonl.gz) |
| 投影兼容修复前扩大验证 | [manifest](before-dual-projection-validation/manifest.json) | [summary](before-dual-projection-validation/summary.json) | [JSONL 压缩归档](before-dual-projection-validation/runs.jsonl.gz) |
| 投影兼容修复前固定步长 | [manifest](before-dual-projection-baseline-control/manifest.json) | [summary](before-dual-projection-baseline-control/summary.json) | [JSONL 压缩归档](before-dual-projection-baseline-control/runs.jsonl.gz) |
| 首轮自适应探索 | [manifest](initial-adaptive/manifest.json) | [summary](initial-adaptive/summary.json) | [JSONL 压缩归档](initial-adaptive/runs.jsonl.gz) |

| 主题 | 文件 |
| --- | --- |
| 最终程序与源码身份 | [identity.json](identity.json)、[源码哈希](source-sha256.json)、[源码快照](source.zip) |
| 投影兼容修复前代码身份 | [identity](before-dual-projection-identity.json)、[源码哈希](before-dual-projection-source-sha256.json)、[源码快照](before-dual-projection-source.zip) |
| 硬件与依赖 | [hardware.json](hardware.json)、[依赖锁快照](dependencies.lock.json) |
| 最终构建与回归 | [构建日志](build.log)、[CTest 日志](ctest.log)、[独立链接命令](pdlp-link-command.txt) |
| 真实回归失败过程 | [新测试构造错误](initial-test-failure.log)、[有限负间隙错误](bound-regression-failure.log) |
| 投影兼容修复前通过日志 | [构建日志](before-dual-projection-build.log)、[CTest 日志](before-dual-projection-ctest.log) |
| 未采用的重启探索 | [逐次数据](restart-exploration.json)、[隔离源代码与诊断](restart-exploration-source.zip) |
| 归档完整性与耗时汇总 | [archive-audit.json](archive-audit.json) |
