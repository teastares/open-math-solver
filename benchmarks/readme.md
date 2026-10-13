# 公开 LP 输入索引

[返回项目总索引](../readme.md) · [正式评估口径](../knowledge/experiments/2026-10-11-求解器基线与扩展性实验计划.md)

| 主题 | 权威来源 |
| --- | --- |
| 公开 Netlib 归档版本、地址和逐模型校验值 | [netlib.lock.json](netlib.lock.json) |
| 数据下载与准备 | [fetch_netlib.py](../scripts/fetch_netlib.py) |
| 既有开发集 | [development.txt](development.txt) |
| Netlib 完整清单 | [netlib-all.txt](netlib-all.txt) |
| 排除既有开发模型后的验证清单 | [netlib-validation.txt](netlib-validation.txt) |
| 双引擎交替重复对照工具 | [compare_lp.py](../scripts/compare_lp.py) |
| 现代公开 LP 开发冒烟版本、来源与哈希 | [modern-lp-smoke.lock.json](modern-lp-smoke.lock.json) |
| 现代开发冒烟下载与路径列表 | [fetch_modern_lp.py](../scripts/fetch_modern_lp.py)、[modern-lp-smoke.txt](modern-lp-smoke.txt) |

现代输入来自 [公开 LP 测试目录](https://plato.asu.edu/ftp/lptestset/)，名单在首次求解前按来源、格式与规模冻结，用于开发，不作为最终保留集。运行 `python scripts/fetch_modern_lp.py` 可核对并复用缓存；如环境需要指定可信证书，使用 `--ca-file`，不关闭 HTTPS 校验。生成数据放在 `benchmarks/data/modern-lp-smoke/`，版本以锁文件为准。

输入列表使用相对于列表文件所在目录的路径，不按求解结果修改。Netlib 输入来自 [COIN-OR 的 Data-Netlib 仓库](https://github.com/coin-or-tools/Data-Netlib)，原始集合说明见 [Netlib LP 数据页](https://netlib.org/lp/data/)。数据本身不加入产品二进制；版本和内容校验统一由锁文件维护。
