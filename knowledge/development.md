# 开发与运行说明

[返回项目总索引](../readme.md) · [项目概况](project.md) · [架构与研发路线](decisions/2026-10-11-通用求解器架构与研发路线.md)

## 当前实现

当前有三个 CPU 求解路径：默认 `highs` 支持 LP/MILP；实验性 `pdlp` 使用自主 PDHG 内核，实验性 `ipm` 使用自主稀疏内点法求解连续 LP。自主部分还包括稀疏模型表示、候选解与对偶数值复核、后端接口、命令行和基准运行器。当前没有自研 simplex、分支割、GPU、跨机器通信或芯片实现。

`pdlp` 支持 `baseline`、`adaptive`、`subspace`、`halpern` 与 `halpern_reflected`，默认仍为 `adaptive`。当前自主内点法已验证并保留的版本为 `ipm-cpu-0.53.0`；工作区候选源码标识以 [ipm_version()](../src/ipm.cpp) 为准，它不是主程序的项目版本号。具体算法取舍见[自主原始对偶设计](decisions/2026-10-12-自主CPU原始对偶原型.md)；`ipm` 的数值方案、信息复用及未实现能力见[自主内点法设计](decisions/2026-10-12-自主稀疏内点法.md)。历次实际构建与测量见[实验索引](experiments/readme.md)，源码版本不代表已经通过新一轮验证。

| 入口或模块 | 权威实现 |
| --- | --- |
| 构建目标与 C++ 标准 | [CMakeLists.txt](../CMakeLists.txt) |
| 第三方版本、地址与 SHA256 | [dependencies.lock.json](../dependencies.lock.json) |
| 自主 CSC 模型、连续与整数变量 | [model.hpp](../include/solver/model.hpp)、[model.cpp](../src/model.cpp) |
| 验证容差与候选解检查 | [verification.hpp](../include/solver/verification.hpp)、[verification.cpp](../src/verification.cpp) |
| 原始/对偶数值最优检查 | [lp_optimality.hpp](../include/solver/lp_optimality.hpp)、[lp_optimality.cpp](../src/lp_optimality.cpp) |
| 自主 LP 算法与预算 | [pdlp.hpp](../include/solver/pdlp.hpp)、[pdlp_options.hpp](../include/solver/pdlp_options.hpp)、[pdlp.cpp](../src/pdlp.cpp) |
| 自主内点法与预算 | [ipm.hpp](../include/solver/ipm.hpp)、[ipm_options.hpp](../include/solver/ipm_options.hpp)、[ipm.cpp](../src/ipm.cpp) |
| HiGHS 固定算法与 crossover 对照 | [highs_options.hpp](../include/solver/highs_options.hpp) |
| 求解选项、结果与可替换接口 | [backend.hpp](../include/solver/backend.hpp) |
| HiGHS 适配、状态和界处理 | [highs_backend.cpp](../src/highs_backend.cpp) |
| 命令行和 JSON 字段 | [main.cpp](../src/main.cpp) |
| 基准参数、统计口径与原始记录 | [benchmark.py](../scripts/benchmark.py) |
| 共同精度的交替重复 LP 对照 | [compare_lp.py](../scripts/compare_lp.py)、[公开输入索引](../benchmarks/readme.md) |

## Windows 安装与构建

从仓库根目录在 PowerShell 中执行：

```powershell
./scripts/bootstrap-windows.ps1
./scripts/build-windows.ps1 -Jobs 4
```

[安装脚本](../scripts/bootstrap-windows.ps1)下载便携 Python、Clang 工具链、CMake 和 Ninja，校验下载归档后放入仓库 `.tools`；不会写入系统 PATH。[构建脚本](../scripts/build-windows.ps1)临时设置自身进程的 PATH，先配置、构建 [OpenBLAS 依赖项目](../third_party/openblas/CMakeLists.txt)，安装到 `build/openblas/install`，再配置主项目、获取并校验其余源码、编译并执行回归测试；`-SkipTests` 仅跳过末尾 CTest。首次安装和首次配置需要连接依赖锁中的下载站点；已有下载及构建缓存可复用。

OpenBLAS 必须先于主项目配置完成安装，因为 SuiteSparse 在配置阶段检查 BLAS/LAPACK 符号。当前依赖项目构建单线程静态库，提供 CBLAS、Fortran ABI 的 BLAS 和由 C 源码编译的实数/复数 LAPACK，不需要 Fortran 编译器；原来的仅 CBLAS、无 LAPACK 产物不满足这个要求。直接调用主项目 CMake 时，也应先完成该依赖项目的构建；非默认安装位置通过 `SOLVER_OPENBLAS_ROOT` 指定，主项目在静态库不存在时明确拒绝配置。配置与编译阶段的详细参数以两处 CMake 定义及构建脚本为准。

依赖构建会按[依赖锁](../dependencies.lock.json)自动应用[OpenBLAS本地补丁](../third_party/openblas/patch_dgemm_stack.cmake)，校验原源码、脚本和修补结果；未知源码拒绝，已修内容幂等跳过。该补丁修正共享乘法内核的缓冲容量与Windows逐页触碰，安装时附带[本地说明](../third_party/openblas/NOTICE.txt)。外部自行提供的库不因此自动获得这项修复；复现当前实验应使用锁定构建及公共BLAS回归，具体证据见第48轮实验正文。

生成 `build/release/solver.exe`。环境与历次结果记录在[实验索引](experiments/readme.md)。其他操作系统和编译器尚未实际验证。构建缓存、便携工具和日常实验输出由 [.gitignore](../.gitignore)排除，交付产物应另行保留。

## 单个模型求解

```powershell
./build/release/solver.exe --help
./build/release/solver.exe --version
./build/release/solver.exe solve tests/fixtures/lp_min.mps
./build/release/solver.exe solve tests/fixtures/milp_min.lp --time-limit 30 --threads 1 --seed 0 --mip-gap 0.0001
./build/release/solver.exe solve tests/fixtures/lp_min.lp --backend pdlp --time-limit 5 --iteration-limit 200000
./build/release/solver.exe solve tests/fixtures/lp_min.lp --backend pdlp --pdlp-mode baseline
./build/release/solver.exe solve tests/fixtures/lp_min.lp --backend pdlp --pdlp-mode subspace --pdlp-history-size 8
```

参数默认值以[求解选项](../include/solver/backend.hpp)及 `--help` 为准。`--iteration-limit`、`--scaling-passes` 用于 `pdlp` 或 `ipm`，按最终后端归属，与参数顺序无关。`--pdlp-mode`、`--pdlp-history-size` 只用于 `pdlp`；历史容量为 0 时关闭子空间搜索，其他模式不使用历史容量。`--highs-solver`、`--highs-crossover`、`--highs-presolve` 只用于 `highs`，MILP 拒绝非默认配置。自主路径目前只支持 `--threads 1`，随机种子不改变其确定性迭代；`--mip-gap` 不控制连续 LP 的停止条件。

`--ipm-objective-scaling on|off` 仅用于 `ipm`，默认 `on`。它独立控制行列缩放结束后的目标标量归一化：开启时沿用最大绝对成本与1的较大值，关闭时保留当前成本、目标恢复系数为1；行列均衡、原模型和终验不变。报告 `options.ipm.objective_scaling` 为实际布尔值。基准工具支持同名候选选项及独立的 `--reference-ipm-objective-scaling`；仅关闭侧标签追加 `:objective_scaling=off`，默认与显式开启等价，显式请求与实际布尔回报不符计失败。此开关不是依据输入身份的自动策略，测量与是否推广由[第五十一轮实验](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)维护。

```powershell
./build/release/solver.exe solve tests/fixtures/lp_min.lp --backend ipm --time-limit 5
./build/release/solver.exe solve tests/fixtures/lp_min.lp --backend pdlp --pdlp-mode halpern_reflected
./build/release/solver.exe solve tests/fixtures/lp_min.lp --backend highs --highs-solver ipm --highs-crossover off --highs-presolve off
```

JSON 的 `options.pdlp`、`options.ipm`、`options.highs` 分别记录对应后端选项，其余为 `null`；内点法另报告迭代、数值分解尝试、迭代改进等计数。第三方稀疏线性代数组件会在构建过程中按依赖锁获取并校验，Windows 可复用 `.tools/downloads` 缓存；SuiteSparse 部分构建 AMD、COLAMD、CHOLMOD 与 SuiteSparse_config，另构建 QDLDL，并使用 Eigen 头文件构建可选 SparseLU 和标量正规方程 Cholesky。Eigen 路径禁用内部并行，仅启用 MPL2 兼容部分，不依赖外部 BLAS。CHOLMOD 与[隔离构建的 FactorHiGHS 内核](../third_party/factorhighs/build.cmake)共用上述单线程完整 OpenBLAS；参考 HiGHS 仍关闭 HIPO，未因此启用其 LP 内点算法。CHOLMOD 构建关闭 OpenMP、CUDA、Partition 等模块；首次构建会增加下载和编译时间，CMake 最低版本以主构建定义为准。

当前构建显式启用 CHOLMOD 的 GPL Supernodal 模块，其上游许可为 `GPL-2.0-or-later`；各模块分别授权，不能将本配置描述为仅使用 BSD 或 LGPL。运行时选择其他分解器不会移除构建中的这项依赖。组件范围与声明见 [CHOLMOD NOTICE](../third_party/cholmod/NOTICE.txt)；安装步骤保存 CHOLMOD、SuiteSparse、AMD、COLAMD、OpenBLAS 和 LAPACK 的相关许可文件，具体路径以 [CMake 安装定义](../CMakeLists.txt)为准。

`--ipm-singleton-elimination on|off` 显式选择单非零不等式的线性代数消元，同后端消融可指定 `--reference-ipm-singleton-elimination`。`--ipm-dynamic-regularization on|off` 控制已知符号的局部主元正则，参考侧用 `--reference-ipm-dynamic-regularization` 独立设置。`--ipm-kkt-equilibration on|off` 对每步实际 KKT 作一次对称均衡，参考侧用 `--reference-ipm-kkt-equilibration`。三项默认均关闭。报告的 `ipm_kkt_dimension` 是实际分解维数，`ipm_eliminated_singletons` 为被消去行数，`ipm_regularized_pivots` 对自有 QDLDL 动态正则包括失败尝试中已修正的主元数；factorhighs 只报告第三方成功分解返回的非零对角扰动数，不声称覆盖其失败分解内部的未返回记录。实验结果及未启用默认的原因见[迭代记录](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。

`--ipm-linear-solver qdldl|sparselu|normal|qdldl_extended|factorhighs|normal_augmented|normal_multifrontal|normal_extended|normal_refined|normal_cholmod|normal_cholmod_aat` 显式替换分解器或选择结构预算回退模式，参考侧用 `--reference-ipm-linear-solver`；默认 QDLDL。SparseLU 保留同一 AMD 排序与正则 KKT，增加数值行选主元，复用模式分析和多 RHS 因子；不将 LU 主元符号解释为惯性，拒绝与局部动态正则同时启用。normal 通过正规方程 Cholesky 解同一正则 KKT，当前要求开启单行消元、关闭局部动态正则；原方程恢复与严格残差验收保留。

normal_augmented 先尝试建立 normal 符号结构，仅在适配器的显式结构预算异常时切换一次到 factorhighs；同样要求开启单行消元、关闭局部动态正则。所有已发生的转换、排序和分析计入同一预算，不重置计时；输入异常、分配失败、时间中断及后续数值分解失败不触发该切换。显式 normal 仍在超限时报告失败。normal_multifrontal 保留同样的 Gram 装配、AMD 和完整方向恢复，但用已有 FactorHiGHS 分块 LDL 分解正定 Schur 矩阵，并在相同正规方程结构预算异常时回退增广形式；分块内核自身的额外容量异常直接报告，不转换为正规方程预算异常。该分块 Schur 路径每次分解按已形成矩阵的正对角做对称均衡，缩放 RHS、求解后恢复 Schur 坐标；每次重新生成尺度，额外扫描和恢复计时，有限值或非零下溢检查失败会拒绝数值分解/方向。标量 normal 路径保持原样。`ipm_structural_fallbacks` 记录已启动的结构切换次数，IPM 未切换时为零；`ipm_factor_backend` 记录实际建立符号结构的内核，尚未建成时为 null。维数和因子计数按该实际内核解释，不能由请求的模式名称推断。

normal_extended 在相同正规方程结构与 AMD 排列下，把已有 long double Gram 累加结果保留到 Cholesky 分解、Schur 回代和原始方向恢复完成，再将完整输出转换为 double。输入 KKT、外层向量和共同终验仍是原接口；不同时启用分块 BLAS Schur 内核，也不引入新正则或搜索规则。平台必须提供更多 long double 有效位，否则显式拒绝；本地格式不代表任意精度。它与 normal_augmented 一样只在专门结构预算异常时回退到 factorhighs，其他错误不回退；实际内核字段分别为 normal_extended 或 factorhighs。扩展矩阵与因子会增加存储和算术成本，现有结构预算按项数计算，不是字节硬上限；成本进入同一时间预算。

normal_refined 保留 double 标量 Schur 分解，利用已有 long double Gram 累加结果和约化右端计算真实 Schur 残差，最多复用因子做两次附加三角求解。Schur 候选和原始方向恢复保留 long double；仅当真实残差无穷范数严格下降才保留校正。候选不可可靠表示或不再下降时保留已有候选，预算异常继续传出；完整 KKT 与原模型仍用原规则验收。它同样要求平台有额外有效位、开启单行消元，且只在显式结构预算异常时回退 factorhighs。校正无法修复一个尚未成功分解的 double 矩阵，也不保证真实 KKT 精度改善。

normal_cholmod 显式选择 CHOLMOD 超节点 Cholesky：保持 normal_augmented 的显式 Gram、long double 装配后转 double、既有 AMD 排列、约化 RHS 与完整方向恢复，只替换数值分解和回代内核。适配器对已排列的矩阵使用 NATURAL、关闭 postorder、强制 supernodal，并复用符号结构和回代工作区；不加入 normal_multifrontal 的额外 Schur 对角均衡或新的主元扰动。该模式要求单行消元开启、局部动态正则关闭，候选与参考独立配置；仅显式结构预算异常回退 factorhighs，数值失败、内存失败和时间预算异常不触发此回退。实际内核字段为 normal_cholmod 或 factorhighs，外层方法、线性精度规则和原模型终验仍按各自选项执行。它没有新增默认选择规则，也不意味着已经测得性能收益。

normal_cholmod_aat 是第63轮保留的显式矩形输入路径，配置条件与结构回退语义同normal_cholmod，实际内核标签为normal_cholmod_aat或factorhighs。其数学表示、数值与存储边界由[适配器](../src/normal_cholesky.cpp)和[第63轮实验正文](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)维护；该选择不改变默认qdldl，也不改变原KKT、HSD或原模型验收。参考侧可独立使用同名选项值。

```powershell
./build/release/solver.exe solve tests/fixtures/lp_min.lp --backend ipm --ipm-linear-solver normal_cholmod --ipm-singleton-elimination on
```

`ipm_schur_residual_products`、`ipm_schur_refinement_steps`、`ipm_schur_refinement_acceptances` 分别记录已启动的扩展精度 Schur 残差乘积、附加因子回代和实际保留的校正；累计包含后来失败或中断的工作，重新分解不清零。这些与完整 KKT 的 `ipm_refinement_steps` 分开计数，不是其子集；接受次数属于 Schur 回代次数，不能再当额外回代叠加。其他 IPM 内核为零，非 IPM 后端为 null。当前内部每次 Schur 求解的校正预算固定为 2；所有额外扫描、分配、类型转换和算术计时，不能称为免费信息。

qdldl_extended 使用锁定 QDLDL 的同一源码、独立符号构建，将分解与回代数值类型改为 `long double`；输入矩阵、正则项、外层 double 向量及原方程验收不变。平台必须提供比 double 更多的有效尾数位，否则显式拒绝；不保证所有平台的 long double 具有相同精度。当前同样拒绝与局部动态正则组合，额外算术、因子内存及类型转换均计入成本，因子非零数口径与 QDLDL 相同。

factorhighs 只复用锁定 HiGHS 中的多波前 LDL 内核，使用相同的 AMD 后 KKT；适配器转为下三角 CSC，并复用超节点符号分析和每步数值因子。数值分解或后续检查未完成时，下次重试重建第三方数值工作区，保留已验证的符号结构；成功路径继续复用工作区。上游块内选主元和小主元扰动保留，因此即使自有 `--ipm-dynamic-regularization off`，也可能报告数值扰动；该开关仍只控制 QDLDL，factorhighs 拒绝其 on 值。调度改为同步执行，BLAS 独立编译为单线程并在运行时选择 CPU 内核；不会初始化或改变 HiGHS 线程池。第三方副本仅作接口及扰动记账适配，详见[来源说明](../third_party/factorhighs/NOTICE.txt)。该模式不调用 HiPO 的 LP 迭代或预求解，原始 KKT 和 LP 终验保持不变。

`ipm_factor_nonzeros` 对 QDLDL 记录含单位对角的 L 结构数，对 SparseLU 记录最后一次成功分解的 L/U 结构数之和，包含两组对角及超节点内保留零项；normal、normal_extended 和 normal_refined 记录 Cholesky L 含对角的结构数，`ipm_kkt_dimension` 此时为实际正规方程维数，纯变量界消元后可为零。factorhighs 记录最后成功分解的符号 L 结构数，包含对角和超节点合并引入的保留零项。normal_multifrontal 记录 Schur 矩阵的实际分块 L 存储和维数，小主元正则计入 `ipm_regularized_pivots`，不将正则后的线性解自动视为原 KKT 解。

normal_cholmod 与 normal_cholmod_aat 的 `ipm_factor_nonzeros` 使用 CHOLMOD 的 `xsize`，即超节点因子数值数组的存储项数，含合并保留的零项和未使用的块内上三角占位，不等于数学非零数；`ipm_kkt_dimension` 为其 Schur 维数。符号分析得到 `xsize` 后、首次分配数值因子前检查既定项数预算，超限属于显式结构预算回退。统计在建立符号结构时即可取得，不证明数值分解成功；转为 factorhighs 后按其实际口径解释。所有后端的计数都不能直接当作内存字节或进程峰值比较。

normal_cholmod的适配路径保留原上三角符号分析，并缓存固定下三角结构及数值源槽映射；每次分解复制当前数值，省去上游重复转置的结构重建。缓存空间不计入因子xsize，因此不能用该字段声称总内存没有变化。实现与预算边界由[适配器](../src/normal_cholesky.cpp)定义，新增存储与实际得失由第49轮实验正文维护；该缓存说明不适用于normal_cholmod_aat矩形输入。

normal各显式Gram路径共用的装配在long double指数范围足够时，每个right系数先计算一次`b/diagonal`，再按原left次序累加`a*scaled_b`；平台范围不足则保留原表达式。范围条件及有限性、非零下溢、累加和转换检查由[适配器](../src/normal_cholesky.cpp)定义，未缓存跨分解的数值，也不新增CLI。舍入次序可能改变外层轨迹，完整KKT与原模型验收保持；第53轮的实际对照、保留依据和性能限制由[实验正文](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)维护，其显式目标归一化off配置不改变全局默认on。normal_cholmod_aat不执行这段我方数值Gram累加。

在另一个更严格的编译期格式与指数范围条件下，Gram装配利用已检查的有限输入、正对角和贡献数量上限，省略被这些条件蕴含的单项有限、非零下溢及部分和有限检查。此改动不再改变浮点表达式或累加顺序；格式或预算条件不满足时保留原逐项检查，输入、时间预算、最终转换和因子完整检查继续执行。实际有符号相消部分和允许为零或次正规，只需有限；范围证明和第54轮完整验证由实验正文维护，不扩展为其他循环可无条件省略数值检查。

第55轮曾在冻结v58候选中尝试严格Krylov耗尽后一次切换同一正则增广矩阵的SparseLU。候选未获得新增覆盖，未通过预先固定的扩集门槛，已撤回本轮五份源文件，当轮恢复为v57。该候选的两个数值回退计数字段只存在于归档程序及报告中，不是当前CLI接口；研究、负结果和恢复验证见[第55轮实验正文](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。

第56轮v59仅在独立副本采集失败常数KKT的实际中间量，未增加生产选项或改变已选择版本。构建、四组程序对照、精确误差分解及适用边界统一见[实验正文](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)；诊断环境变量只属于该归档程序，不是当前CLI接口。

第57轮v60的宽方向实验未推广。候选的homogeneous_extended选项及两项扩展Krylov计数只存在于冻结程序和源码包，当前内核已恢复此前选择；仅保留CLI测试的平台探测隔离修复。候选结果、恢复后回归和当前来源边界见[第57轮实验正文](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。

第58轮v61继续在冻结v60副本采集常数方向恢复事件，生产接口与数值内核未改。[构建](experiments/data/2026-10-12-halpern-ipm/v61-build.py)、[运行](experiments/data/2026-10-12-halpern-ipm/v61-run.py)及[离线分析](experiments/data/2026-10-12-halpern-ipm/v61-analyze-hsd.py)的使用边界和实际结论见[第58轮正文](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。该诊断的环境开关不属于当前CLI能力。

normal 的 Gram 形成工作与 S/L 结构容量由[适配器](../src/normal_cholesky.cpp)中的显式预算限制，超限报告失败，不自动更换后端；上述带结构回退的模式另按各自约定处理。factorhighs 的[适配器](../src/multifrontal_ldl.cpp)另限制输入/预测填充及超节点工作区估计。这些是实验容量边界，不是进程内存硬上限。第三方库调用前后检查预算，不保证库内部硬中断。

`--ipm-method mehrotra|homogeneous` 选择自主内点方法，默认 mehrotra，继续原预测校正循环。homogeneous 是独立的连续 LP 齐次嵌入实验：增加正 τ、κ，把工作模型目标间隙并入联立 Newton 方程；每次分解求一次常数右端，预测和校正分别求方向右端并共享该常数解。接口与标量消元见[齐次接口](../include/solver/detail/ipm_homogeneous.hpp)，具体方程、来源和实测得失见[第四十一轮](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。

homogeneous 支持 strict 或下述 homogeneous_scaled、homogeneous_repair，要求零方向重试/历史/中央校正和关闭非精确预筛选；原分解器前置条件、固定变量处理和缩放选项仍适用。共同步长包含不等式 s/z 及 τ/κ，等式对偶保持自由；先把规范坐标除 τ，再恢复原固定常量及行对偶，最终验算不放宽。更新在完整有限性、正性及预算检查后一次提交，不用标量下限掩盖退化，不由 τ/κ 比例宣称不可行或无界。尚未实现完整证书、构基或 crossover。

`ipm_homogeneous_constant_solves` 和 `ipm_homogeneous_direction_solves` 在实际启动相应 KKT 求解前计数，失败的已启动工作保留；初始化辅助求解不计入这两项，但分解/精化仍进入原有统计。strict 完整外层迭代各为1和2次，其内部精化可能额外回代；homogeneous_scaled 则按每次直接回代计数，方向计数包含完整IR校正。因此它们不能跨精度模式当作相同的总浮点工作。`ipm_homogeneous_tau/kappa` 记录最近完整提交的内部点，可能与报告中保留的较早可行解不同。齐次初始化前两标量为 null，普通 IPM 方法计数为整数零且标量 null，非 IPM 四项均 null。比较工具的 `--reference-ipm-method` 独立设置参考方法；显式方法必须与实际报告一致。

`--ipm-linear-accuracy strict|backward|inexact|blockwise|blockwise_weighted|blockwise_forcing|blockwise_forcing_weighted|adaptive_regularization|primal_dual_regularization|curvature_regularization|full_newton|bounded_newton|homogeneous_scaled|homogeneous_repair` 控制内部线性方向验收；默认 strict，backward 额外允许分量后向误差达到单位舍入量的方向。inexact 在旧严格条件未过时，按实际恢复后原始/对偶方向误差与各自外层残差的比例验收；初始化仍严格，不引入互补量或右端大小的绝对地板。`ipm_backward_error_acceptances`、`ipm_inexact_acceptances` 分别只统计旧严格条件未过而对应替代条件通过的次数。所有模式均保留相同的原模型终验。

homogeneous_repair 仅用于 homogeneous，初始化与正常方向沿用strict。仅当标量消元的分母有限且非正、方向点积有限时，才用同一KKT因子对常数右端[-c;b]追加一次blockwise求解；它同时要求原全局门槛和两块门槛，内部沿用有界IR/Krylov预算，不是只做一次三角回代。候选常数还必须通过原标量消元和预算检查才提交，同轮预测/校正共享修正结果及一次尝试额度。不重新分解，不更换正则层级，不夹取分母；其他数值错误、预算和内存异常正常传播。

`ipm_homogeneous_constant_repairs`在实际启动该额外常数求解前累加，同时计入`ipm_homogeneous_constant_solves`；内部IR/Krylov工作进入原有统计。`ipm_homogeneous_constant_repair_successes`在修正常数经标量检查提交后累加，表示局部恢复成功，不等于当前方向或最终LP达标。计数满足successes≤repairs≤constant_solves；其他IPM模式两字段为整数零，非IPM为null。候选与参考精度独立配置，新模式的成功报告要求两字段合法，旧strict/scaled归档无需补字段。研究、触发依据与实测结果见[第四十七轮](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。

homogeneous_scaled 仅用于 homogeneous，是独立的缩放齐次方向政策。以 `d=sqrt(kappa/tau)`、`W=sqrt(s/z)`、`lambda=sqrt(s*z)` 构造缩放，每次分解直接求一次常数右端 `[-c/d;b/d]`，用 `1+||W*v_z||²` 作近似标量消元；它不是把原失效分母取绝对值或加下限。完整五块乘积从实际交付的物理 double 方向及原 G/s/z/τ/κ 重新形成，间隙、锥互补、标量互补分别除 d、lambda、sqrt(tau*kappa)，包含近似常数解与恢复舍入的缺陷。

每个预测或校正方向做初次直接回代加最多三次完整IR，仅保留五块最大归一残差严格降低的有限候选。每块门槛固定为本方向初始缩放右端的 `1e-12+1e-10*||rhs_block||∞`，用于提前停；有限方向未达到内层门槛仍可返回并如实统计，不等于LP成功。不在每次IR内嵌旧Krylov，初始化仍是原strict；原分解、外层中心化/步长与原模型终验不变。回调主动异常、初方向非有限和时间预算继续传播，可选候选自身不可表示时保留已经验算的最佳方向，不在验算后重新计算ds。

`ipm_homogeneous_scaled_products` 记录完整五块乘积实际启动次数，`ipm_homogeneous_scaled_refinement_steps` 记录完整IR校正直接回代的启动次数，`ipm_homogeneous_scaled_inexact_returns` 记录最终有限但未达内部门槛的方向返回次数。该模式的完整IR同时计入总 `ipm_refinement_steps`，两项不可重复相加。方向计数包含每次IR，改进次数不超过方向启动次数；非精确返回不超过方向启动或乘积次数。三字段在其他IPM模式为整数零，非IPM为null；显式请求新模式的成功报告必须具有合法统计。候选与参考精度、方法独立配置，未显式请求新模式时旧归档报告保持兼容。实际得失见[迭代记录](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。

blockwise 在 Newton 方向中额外要求完整 KKT 两块各自满足 `1e-12+1e-10*||rhs_block||∞`，两个初始化辅助投影仍用原 strict 门槛；它不允许以前述替代条件绕过分块检查。blockwise_weighted 使用同样的验收，并把两块门槛的倒数用于 Krylov 搜索的左尺度，预条件器右端先恢复原尺度，返回前重新计算未缩放原 KKT 缺陷。权重在本次方向求解期间固定，不由模型名称或最优值决定。`ipm_weighted_krylov_steps` 是总改进计数中的加权步骤子集，也包含下述 full_newton 的三块加权 Arnoldi 步骤，其余未加权模式为零；向量缩放、候选重算与检查成本全部计时。blockwise_forcing 在 Newton 阶段改用 `1e-12+0.1*||rhs_block||∞`，仍与全局 strict 条件取交集，初始化沿用 strict；此模式不加权，也不能经 backward/inexact 替代条件绕过。blockwise_forcing_weighted 使用完全相同的真实门槛，将全局与对应块门槛的较小者用于搜索尺度；沿用加权步骤计数、初始化及预算。上块 RHS 是当前对偶残差的负值，下块还包含互补消元项，因此这里不声称保证两个实际外层残差的相对进展。这些均为可关闭实验，具体得失见[迭代记录](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。

full_newton 是显式选择的完整未消元 Newton 搜索，默认仍 strict。方向按 `(dx,dz,ds_I)` 存储，I 仅包含不等式行，E 将 I 分量嵌回全部 m 行；真实算子为 `(Gᵀdz, Gdx+E ds_I, Z_I ds_I+S_I dz_I)`，等式行没有松弛变量或互补方程。预测/校正分别使用 RHS `(-rd,-rp,h_I)`，三块各按 `1e-12+1e-10*||rhs_block||∞` 独立验收。直接扫描原 G、s、z 并以 long double 累加完整乘积，不用舍入后的 H=s/z 反推真实原始或互补块。它替换外层方向的消元 K0 验收问题，不能视为原全局门槛的简单收紧；原模型最优性终验保持不变。

[完整 Newton 接口](../include/solver/detail/ipm_full_newton.hpp)使用现有 KKT 因子作右预条件：将任意三块右端缩为 `(b_d,b_p−E(b_c/z_I))`，直接回代后按 `(b_c−s_I*dz_I)/z_I` 恢复不等式 ds；等式行不除以任意符号乘子。先做一次直接预条件解，最多两次改进，再最多三个周期、每周期 `min(32,n+m)` 列 FGMRES，不在预条件器中嵌套旧 Krylov。搜索按本次三个门槛的倒数左缩放，预条件器右端恢复物理尺度；保存每列实际预条件方向，两遍正交化。估计残差只触发候选构造，每个返回方向必须通过重新计算的物理三块残差。完整向量变长，会增加存储、扫描、正交化和恢复成本；窗口列数不增加不意味着成本相同。

初始化的两个辅助求解继续原 strict 流程。full_newton 支持方向重试预算 0..8，重试仍受已有八层正则网格限制；history 正预算和附加中央校正暂拒绝，非精确预筛选仍只允许 inexact。原单行消元、因子均衡和各分解内核的前置条件继续适用。所有额外工作与失败计入同一数值时限，原模型终验不放宽；当前能力不构成收敛或性能改善保证。

`ipm_full_newton_preconditioner_solves` 在实际启动直接 KKT 回代前累加，包含初始、普通改进及 Arnoldi 的调用；`ipm_full_newton_products` 在真实 J 乘积开始前累加，包含候选验收与 Arnoldi。中断的已启动工作也保留。`ipm_refinement_steps` 累加已完成的额外预条件调用，不含初始调用；其中 Arnoldi 同时计入 `ipm_weighted_krylov_steps`，不能把这些重叠统计再次相加。这两项同时统计 full_newton 与 bounded_newton 的实际工作，在其他 IPM 精度模式为整数零，非 IPM 为 null。比较工具沿用独立 `--reference-ipm-linear-accuracy full_newton`；显式请求须匹配实际字符串报告，拒绝缺失、错误类型或静默回退。实际收益只在[实验正文](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)记录。

bounded_newton 是独立的有界完整 Newton 实验，使用上述同一真实 J、三块固定 RHS 尺度和直接 KKT 预条件器。初始方向加最多三次 IR，只保留真实归一化残差严格更小的有限候选；达标、停滞、可选候选非有限或三次 IR 预算用尽即停止精化，不调用 FGMRES。内层未达门槛也可返回有限方向，仍由既有外层推进并执行相同原模型终验。它没有新增外层试探回滚，也不构成收敛保证；初始非有限和回调主动抛出的数值异常仍失败，时间、内存及非法维数异常继续传播。初始化保持 strict，正 history、方向重试、中央校正以及 inexact 预筛选均不支持。

`ipm_bounded_newton_returns` 只在方向恢复且末尾预算检查通过后增加；`ipm_bounded_newton_inexact_returns` 是其中未满足三块内层门槛的次数，不能计作 LP 成功。两者在其他 IPM 模式为整数零，非 IPM 为 null。完整 Newton 的 P/J 计数沿用上述字段，IR 记入总改进次数，加权 Arnoldi 为零。比较工具支持候选/参考独立选择 bounded_newton，拒绝模式回报不符和非法统计；实际得失见[第四十轮](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。

`--ipm-krylov-history-size 0..8` 设置同一完整 K0 的回收子空间最大维数，默认 0。正值仅支持 strict、blockwise；初始化仍使用原 strict 验收，blockwise 的外层全局与分块条件保持原样。显式 0 兼容所有现有精度模式；正值暂不支持 full_newton、bounded_newton、加权、forcing、backward、inexact 或其他正则模式。它可与已有方向重试和中央校正组合，原 Krylov 每周期至多 32 列、至多 3 个周期的预算不变。

[历史容器](../include/solver/detail/ipm_recycling.hpp)成对保存完整 K0 坐标的方向 U 与算子像 C，用两遍正交化维护 C 的正交性，同步变换 U；退化、非有限或转换下溢的候选不入池，满池按插入顺序淘汰最早项。已按原条件接受的直接/迭代改进方向，以及完成 Krylov 周期的整体修正可以提供候选；不收集每一列 Arnoldi 向量。一个周期内历史保持固定，只在周期完成后更新。历史有效期由同一规范化模型的 G 与 H=s/z 决定：H 的存储值改变时清空；同一 H 下更换正则因子只改变右预条件器，历史可继续供初始化双求解、预测/校正、中央校正或有限方向重试使用，不跨问题复用。

历史非空且进入 Krylov 改进时，在周期开始尝试残差投影。候选必须重新计算真实 K0 残差，并使按原全局/分块门槛归一化的最大缺陷严格下降，才采用投影并启用本周期增广；已满足原方向条件时可直接返回。后续 Arnoldi 对物理预条件方向和算子像同步去除 U/C 分量。投影数值失败保留原点，历史正交化失败保留该次原方向并关闭本周期增广；预算异常仍向外传递，不把失败信息当成接受资格。所有返回方向继续通过原 K0 条件，最终 LP 仍由原模型验算。

入池复用已有真实残差扫描形成的 long double 算子乘积缓存：完整方向使用该乘积，周期整体修正使用两个已算乘积之差，因此入池本身不增加 K0 乘法。缓存差、转回 double 及正交化仍可能舍入或相消，`K0 U≈C` 只是浮点近似，可能使增广失效或退化；真实投影下降与最终原 K0 验收不能省略。历史复制、两遍正交化、投影及候选的真实残差扫描都有成本，全部计入原总预算。历史向量存储规模为 O(Nk)，其中 N 是完整 K0 维数，k≤8；此外保留乘积缓存和临时候选，不将计数或向量规模当作进程内存硬上限。

`ipm_krylov_recycle_projections` 统计历史非空时已启动的周期投影尝试；`ipm_krylov_recycle_acceptances` 只统计投影后直接满足原方向条件并返回的次数，不包含仅下降后继续 Arnoldi 的情况。`ipm_krylov_recycle_vectors` 累计成功入池的向量对，包含后来淘汰或清空的项；`ipm_krylov_recycle_peak_dimension` 为本次求解的最大同时保留维数，不超过 k。IPM 从 0 开始计数，关闭时四项均为 0，非 IPM 为 null；入池或投影接受不代表最终 LP 成功。

比较工具以 `--reference-ipm-krylov-history-size` 独立设置参考侧，分别检查两侧精度许可。报告 `options.ipm.krylov_history_size` 为整数；省略参数时请求元数据为 None、实际默认 0，显式 0 仍核对精确整数类型和值，拒绝布尔数伪装。只有正值在原配置标签末尾追加 `:krylov_history=N`；非 IPM 拒绝任何显式预算，包括 0。此项是默认关闭的实验，实际得失以[实验正文](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)为准，不由接口存在推断性能收益。

adaptive_regularization 保留 strict 的方向验收及初始化，只在外层预测方向已经求出后，用验收时已有的真实 K0 对偶块缺陷决定是否试探更小正则量。目标为 `δ*||dx_aff||∞ ≤ 1e-12+0.1*||rd||∞`；实际缺陷已达目标或本次分解已经从较小正则重试到较大值时，不启动这次额外搜索。沿原十倍网格向下扩展到仍不小于 double epsilon 的最小格点，每个外层至多一次搜索；分解失败仍逐级增加，最高格点不变。成功降低后在后续外层复用较低起点；预测方向重算，校正方向共享新因子。目标仅用于选搜索起点，不能保证新的预测/校正方向达到该块进展条件，也不能代替原模型终验。`ipm_regularization_searches` 计启动次数，`ipm_regularization_reductions` 计成功获得较小正则分解的次数；后者不等于求解收益。IPM 其他模式两项为零，其他后端为 null，全部分解和改进成本继续计入原账本与时间预算。它与局部主元修改开关的作用不同；本实验不改变默认模式。

primal_dual_regularization 沿用 strict 的方向验收，以 `ρ=ε^(3/4)` 和 `δ=ε^(1/2)` 作为缩放前原始、对偶外部均匀正则起点，其中 ε 为 double epsilon。失败时两者仍沿原十倍层级共同增加，每次最多 8 次分解尝试；这是不同量级的静态消融，尚未分别自适应搜索两项，也不复现论文的局部主元修正。初始化同样采用这两个起点。KKT 原始块加 ρ，对偶块使用 `-(H+δI)`；单行消元、因子均衡和完整方向恢复使用同一参数，原 K0 与原模型精度保持原样。

curvature_regularization 保留 strict 方向验收，初始化及每轮分解先检查本轮单行消元产生的原始对角：全部在 double 中有限且严格为正时，额外尝试一次零原始均匀偏移，仍保留旧起点的对偶正则。缺少正曲率时不启动这次额外分解；分解失败后进入旧的 8 级相同正则网格，因此每轮至多 9 次尝试。它不把分解成功视作原方程精度保证；后续方向求解仍按原规则检查，默认不重试方向失败。分解重试复用原符号结构；检查、装配和所有失败成本均计时。`ipm_unshifted_factorizations` 记录已启动的零原始偏移分解次数，是 `ipm_factorizations` 的子集，不代表方向接受或最终求解成功；其他 IPM 模式为零，非 IPM 为 null。

`ipm_primal_regularization`、`ipm_dual_regularization` 报告最后一次已启动数值分解尝试的外部均匀参数，坐标为外部 KKT 均衡前。它们不是浮点加法后每个位置的有效偏移，也不含内核局部主元修正；尚未启动分解或非 IPM 后端为 null。curvature_regularization 的原始参数可为零，primal_dual_regularization 两项有不同量级，其余模式保持原有两项相等的规则，默认仍 strict。

`--ipm-central-corrections 0|1` 设置每个外层迭代的附加中心校正预算，默认 0；报告尝试数与至少一个方向分组接受的次数。当前明确拒绝 inexact、full_newton、bounded_newton 与校正预算 1 组合，避免叠加尚未控制的方向误差。比较工具分别提供同名 `--reference-` 前缀选项。实验方法、收益与限制见[迭代记录](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。

`--ipm-direction-retries 0..8` 设置初始化双求解组及每个外层预测/校正组因数值失败而重启的预算，默认 0。正预算支持 strict、primal_dual_regularization、curvature_regularization、blockwise、blockwise_weighted、full_newton。所有初始化辅助求解仍用 strict；外层前三种模式保留 strict 验收，两种 blockwise 模式保留上文全局 strict 与分块条件的交集，并重新计算真实未缩放 K0 缺陷；full_newton 保留其完整三块 Newton 验收。blockwise_weighted 在每次方向求解时根据该次 RHS 重新生成分块门槛和 Krylov 尺度；重试后重算的校正 RHS 不能沿用旧尺度。重试不放宽任何方向条件，不与 backward、inexact、两种 forcing 或 adaptive_regularization 组合。

普通分解从实际成功层级的下一层继续，曲率零偏移之后从旧层级 0 继续且不再尝试零偏移；最高层级仍为 7，已失败层级不重复。因此一组最多覆盖旧 8 个正则格点，曲率模式另有一次零偏移，不因多次方向重试而重新获得完整网格。

重试重建同一矩阵的因子，并重新计算整组 RHS/预测/中心参数/校正；校正失败不能保留旧预测方向。初始化启用时，两个辅助求解均成功才提交原始点；正松弛平移在重试范围之外。外层点更新和可选中央校正在整组之外，保留原异常语义。只捕获组内数值失败，时间预算、内存、输入与分解装配异常不据此继续；预算仍是从求解开始累计的总时限。`ipm_direction_retries` 统计已经发起的重分解请求，内部跳层分别累计到 `ipm_factorizations`；`ipm_direction_retry_successes` 统计重试后完成方向组的次数，不代表外层点更新或最终成功。关闭预算时两项为零，非 IPM 后端为 null。比较工具提供独立的 `--reference-ipm-direction-retries`，显式 0 也核对报告；非零预算在配置标签末尾显示 `:retries=N`。

`--ipm-inexact-prefilter on|off` 复用已有转置乘积，按保守舍入界提前拒绝必定不满足完整 inexact 检查的方向；其他方向仍按原条件验收。默认关闭，仅 inexact 允许开启，参考侧用 `--reference-ipm-inexact-prefilter`。`ipm_inexact_prefilter_rejections` 记录省去的完整检查次数，预计算与缓存成本仍计时。

`--ipm-fixed-variable-elimination on|off` 在数值迭代前代入输入列界中有限且上下界完全相等的变量，默认关闭；参考侧用 `--reference-ipm-fixed-variable-elimination`。固定变量代入本身保留全部原行，每次原模型验算前恢复固定值和列映射，原目标常数不修改。位移无法有限表示或使非零行区间坍缩时保持原表示。`ipm_eliminated_fixed_variables` 报告实际消除列总数，包含下述可选单行归并后新固定列，可能与输入中可识别固定列数不同；零预算不进行消元，所有检查和转换均计时。

`--ipm-singleton-presolve on|off` 在建立标准形式时，将符合条件的单非零输入行归并到变量界，默认关闭；它与 `--ipm-singleton-elimination` 对每步 KKT 不等式变量的线性代数消元独立。比较工具用 `--reference-ipm-singleton-presolve` 独立设置参考侧；两侧显式 off 也严格核对 `options.ipm.presolve_singleton_rows` 的布尔类型和值。只有 on 在已有配置标签之后追加 `:presolve_singletons=on`，关闭时保留旧标签；非 IPM 拒绝显式开关。

归并扫描位于既有固定变量代入或其可靠性回退之后，仅接受扫描表示中恰好一个原始非零存储项、至少有一个有限行端点的整行。负系数交换上下端点；所有有限端点都必须得到可有限表示的精确商，才能吸收该行。内部[精确商检查](../include/solver/detail/ipm_presolve.hpp)将 binary64 数转换为奇整数尾数与二进制指数，用指数相加和整数整除证明 `a*q=b`，不依赖浮点乘积、FMA 或 long double 的近似结果，也不要求系数为 ±1。平台不满足所需 binary64 表示时跳过此预处理；空行、重复非零项、无有限端点或商不能精确表示的行保留原状。此判定只保证本次端点除法，不扩大既有固定变量位移的精度承诺。

每列通过严格 max/min 归并上下界，等值时优先保留原列界来源，其后按原行顺序保留最先收紧者；不使用近似活跃判断。若得到下界大于上界，整列撤销本次归并并保留相关行，继续旧流程，不新增不可行证书。默认将新得到的相等列界作为一个等式进入标准形式；另有下述显式选项进行一次固定值代入，均不执行级联单行归并。原 `Model`、目标常数和报告中的行列规模不修改。

每侧保留原行编号和系数 `a`。恢复时先把归并列界乘子还原到缩放前的变量界尺度及目标尺度，设其有符号值为 `r`：正值归于下界来源，负值归于上界来源；若来源是原行，向对应行对偶累加 `r/a`，不预存 `1/a`。原列界来源不写入行对偶，未分配贡献的已吸收行保持零；其他保留行照旧恢复。原模型检查器重新计算约化成本、可行性与最优性，恢复不可有限表示或非零值转换下溢时报数值失败。`ipm_presolved_singleton_rows` 仅在标准形式成功建立后记录实际吸收的原始行数；IPM 未执行或未吸收时为 0，非 IPM 为 null，不等于 KKT 消元计数。扫描、精确检查、映射存储和每次恢复都计入原求解预算。

`--ipm-presolve-fixed-elimination on|off` 代入本次单行归并产生的新固定列，默认关闭；开启必须同时开启本侧 `--ipm-singleton-presolve on`，与旧输入固定变量开关独立。仅处理扫描前列界并非有限相等、扫描后得到有限相等上下界的列；输入中已固定但旧开关关闭的列不由此选项处理。只进行一次归并后代入，不重新扫描代入后才成为单非零的行，不级联传播新固定点。所有列都被消除时也先恢复原变量和对偶，通过原模型终验才可报告最优。

新消列只对未被吸收的保留行累计固定值贡献，以 long double 累加行端点位移后转回 double，检查非有限数、非零乘积或结果下溢以及严格行区间坍缩。任一检查失败时，整批撤销本轮新消列，继续保留既有单行归并。此处是浮点可靠性保护，不是精确有理变换；前述端点商的精确证明不覆盖这一步的乘加和位移。原 `Model` 与目标常数不修改，位移、回退和恢复均计入原总预算。

恢复时先还原其他行对偶，再对每个新消除列 `j`，用未修改的原模型计算 `r_j = sense*c_j - (Aᵀy)_j`，其中最小化的 sense 为 1，最大化为 −1。正 `r_j` 选择保存的下界来源，负值选择上界来源；来源是原行时累加 `r_j/a`，来源是原列界时留给原列约化成本，不额外写行对偶。来源行仅含本列及此前已经代入的输入固定列，单次处理保证各新固定列恢复互不依赖。恢复使用原列索引、原系数、原目标和已恢复的原行对偶，不用缩小后问题的约化成本替代；不可表示或非零转换下溢仍报数值失败，最终重新检查完整原模型。

`options.ipm.eliminate_presolved_fixed_variables` 报告实际布尔选项；`ipm_presolved_fixed_columns` 仅在标准形式成功建立后报告实际新消除列数，是 `ipm_eliminated_fixed_variables` 总数的子集，关闭、未执行或回退时为 0，非 IPM 为 null。比较工具以 `--reference-ipm-presolve-fixed-elimination` 独立设置参考侧，并分别检查各自依赖；未请求元数据为 None，显式 off 仍严格核对布尔类型和值。只有 on 追加 `:presolve_fixed=on` 标签后缀，off 保留旧标签；非 IPM 拒绝任何显式 on/off。

`--time-limit` 对 HiGHS 限制后端求解时间，不包含读取与外部验证；对自主路径包含模型准备、缩放、迭代及周期检查，不包含文件读取与进程启动。时间检查在迭代/阶段边界执行，最终诊断可能稍微超出预算；它不是硬实时中断。基准运行器另有进程墙钟上限。零求解或迭代预算报告未完成，不伪造解或界。

标准输出只有一个 JSON 对象，诊断写到标准错误。可通过 `--output <新文件>` 同时保存报告；父目录需已存在，既有文件拒绝覆盖。报告包括实际后端、模型规模、参数、版本、状态、目标值、界、间隙、候选向量、验证结果和分阶段时间。三个后端都可提供 `lp_optimality` 和 `row_dual`。自主路径的 `pdlp_iterations` 表示接受步数，不包含线搜索回退的额外矩阵计算，不能直接当作总工作量。没有有效数值时输出 `null`，不会写入非法 JSON 的 NaN 或 Infinity。

`adaptive`/`subspace` 以及两种 Halpern 模式另提供 `pdlp_acceleration_attempts`、`pdlp_acceleration_accepted` 和 `pdlp_operator_passes`，分别记录历史搜索尝试、接纳和内核稀疏乘次数；初始化、回退和候选重算均计入内核次数，每个正向或转置乘计一次。缩放扫描与独立原模型验算不在此计数内，但计入实际耗时；`baseline`、IPM 与 HiGHS 不提供这组 PDHG 内核计数。历史拟合有额外计算和内存成本，不能只凭接受步数变少就判断加速。

| CLI 退出码 | 含义 |
| --- | --- |
| `0` | HiGHS 报告最优、不可行、无界、空模型，或自主路径通过数值最优检查；状态与验证来源分别标识 |
| `2` | 参数、读取、模型支持范围或运行错误 |
| `3` | 候选解或目标/界一致性检查失败 |
| `4` | 时间/迭代限制、数值困难等未完成状态，包括不可行或无界尚未区分 |

不要仅凭退出码为零认定已取得最优解，应同时读取 `status`、`has_solution` 和 `verification_level`。

## 批量运行与留痕

下面从依赖锁定位便携 Python，不单独维护版本字符串：

```powershell
$dependencyLock = Get-Content dependencies.lock.json -Raw | ConvertFrom-Json
$pythonDirectory = ($dependencyLock.windows_tools | Where-Object name -eq 'python').directory
$pythonExecutable = Join-Path (Join-Path '.tools' $pythonDirectory) 'python.exe'
& $pythonExecutable scripts/benchmark.py tests/fixtures/lp_min.mps tests/fixtures/milp_min.lp --solver build/release/solver.exe --time-limit 10 --threads 1 --seeds 0 1 2
& $pythonExecutable scripts/benchmark.py tests/fixtures/lp_min.mps --solver build/release/solver.exe --backend pdlp --time-limit 5 --iteration-limit 200000
```

运行器会创建新的 `results/` 子目录；也可用 `--output-dir` 指定一个尚不存在的目录。它逐次运行模型和种子，首版不并发启动集群任务。

- `manifest.json`：运行预算、环境、可执行文件哈希和输入列表。
- `runs.jsonl`：每次命令、输入哈希、返回码、原始输出文本、解析报告、墙钟耗时和失败分类；失败不会从样本中删除。
- `summary.json`：各状态及失败计数，以及按预算封顶、对失败施加统一惩罚的墙钟 SGM10。

统计公式与分类以[运行器实现](../scripts/benchmark.py)为准。运行器核对请求/实际后端及显式请求的自主模式，并要求自主 `optimal` 报告具有原始、对偶及间隙检查证据。这个通用 LP/MILP 描述统计不等同于下面的共同精度 LP 对照，也未采集完整 gap 轨迹。报告 `relative_gap` 使用有限 `best_bound`，分母为 `max(1, |目标|, |界|)`；`lp_optimality` 中的间隙来自去常数的估计对偶目标，用于数值检查。HiGHS 的 `--mip-gap` 是另一种后端停止规则，三者不可混用。

公开连续 LP 对照使用独立入口：

```powershell
& $pythonExecutable scripts/fetch_netlib.py
& $pythonExecutable scripts/compare_lp.py --solver build/release/solver.exe --models-file benchmarks/development.txt --pdlp-mode adaptive --repeats 3 --time-limit 5 --threads 1
& $pythonExecutable scripts/compare_lp.py --solver build/release/solver.exe --models-file benchmarks/netlib-validation.txt --pdlp-mode adaptive --repeats 3 --time-limit 5 --threads 1
& $pythonExecutable scripts/compare_lp.py --solver build/release/solver.exe --models-file benchmarks/development.txt --reference-pdlp-mode adaptive --reference-pdlp-history-size 0 --pdlp-mode subspace --pdlp-history-size 8 --repeats 3 --time-limit 5 --threads 1
```

[比较工具](../scripts/compare_lp.py)在运行前冻结列表、程序及逐模型哈希，交替两个后端的先后顺序，并保存每次运行前后的输入和程序哈希。只有双方每次重复均达到共同原始/对偶数值标准的模型，才报告逐模型中位数速度比和测量范围；所有未完成、读取错误、质量不合格和进程超时仍计入全体统计。固定输入的来源与内容以[数据锁](../benchmarks/netlib.lock.json)为准，不根据结果删改列表。

用 `--reference-solver <旧程序路径>` 可比较两份冻结程序，待测侧仍由 `--solver` 指定。不同路径允许相同算法参数，标签增加 `binary=reference` / `binary=candidate`；清单的 `solvers` 保存双方路径与哈希，每条记录按所属程序独立校验。省略此参数时沿用同一程序的算法消融。修改中途任一程序或程序缺失会使该侧相应运行失败，不借用另一侧的哈希。

默认比较 HiGHS 与选定的 PDLP 模式。指定 `--reference-pdlp-mode` 后改为同一程序的两种 PDLP 配置交替消融，参考侧历史容量可用 `--reference-pdlp-history-size` 指定。`requested_variant` 与 `variants` 分组记录模式和容量，通用倍率字段为 `reference_over_candidate`，此时旧的 HiGHS 倍率字段为空，避免将自主方法间的收益误报为超越 HiGHS。已观察过的 Netlib 验证清单可继续作固定回归集，但不再称为全新盲测。

`--candidate-backend ipm` 选择自主内点法；`--reference-backend` 可显式选择参考后端。混合算法保留各自默认迭代预算，参考侧可用 `--reference-iteration-limit`、`--reference-scaling-passes` 单独指定。补充点解对照使用 `--reference-highs-solver ipm --reference-highs-crossover off --reference-highs-presolve off`，主参考仍为默认 HiGHS。显式关闭预求解是为了避开当前版本最大化无基 postsolve 的已观察问题，额外计算成本计入该补充参考；来源和异常回归见[本轮实验](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。配置写入清单，显式参数与实际报告不符的运行计失败。

主指标是子进程结束并收集输出时的墙钟时间，不包括驱动中的 JSON 解析；次指标是 `solve_seconds + verification_seconds`，把自主算法内部停止检查和两方共同验算都计入。原始 `solve_seconds` 会扣除自主验算成本，不能单独用作公平加速比。失败惩罚、预算截断和 SGM10 是预先定义的评分，不能解释为真实耗时；原始耗时另行保留。清单哈希会预读文件，因此测试包含文件缓存预热影响，不能称为完全冷缓存测量。中位数更小也不自动代表统计显著优势。

## 支持范围与正确性边界

首版读取未压缩的 `.mps` 和 `.lp`，支持单个线性目标、目标常数、最小化/最大化、连续/整数/二进制变量，以及双边线性约束和变量界。

二次、多目标、半连续/半整数等扩展不在当前支持范围。读取警告、错误或已识别的有限数据转无穷诊断会使输入被拒绝，包括后端阈值导致丢弃小非零系数或放大有限边界的情况。唯一可重试的通知是自由格式 MPS 检测到名称内空格而切换固定格式：新读取器显式按固定格式重读，第二轮仍须无告警、无错误或数据替换；计入两次读取时间，不按模型名称分支。这些诊断识别依赖锁定的 HiGHS 版本，升级时必须复核源码并运行相关回归样例。当前没有独立 LP/MPS 解析器，因此语法解析仍依赖 HiGHS。

在 Windows 上，HiGHS 文件读取器使用系统窄字符路径。不能无损转换的输入路径会明确报错；使用 ASCII 路径最稳妥。报告中的路径采用 UTF-8。

候选验证器根据读取后的、求解前的模型复核行约束、变量界、整数性和目标值，并检测输入结构、非有限数据及累加溢出。详细公式及默认值以[验证接口](../include/solver/verification.hpp)为准。HiGHS 连续 LP 候选也执行共同的原始/对偶检查；其声称最优但共同检查未通过时报告 `verification_failed`，不能计入性能达标。HiGHS 的不可行性、无界性、MILP 最优性与全局界仍来自后端；其 LP 最优界是后端声明的复述。

自主路径在原始坐标额外检查行对偶符号、约化成本残差和去目标常数后的间隙。只有全部通过才报告 `optimal` 与 `primal_dual_residuals_checked`。`row_dual` 使用规范化最小化问题的符号：正值对应行下界；即使原模型是最大化也遵循这一约定。HiGHS 与 PDHG 在验算前将有限行乘子投影到合法行对偶域，再独立重算残差和间隙；非有限乘子不会被投影隐藏。IPM 从非负锥及自由等式乘子恢复原行对偶，再直接严格验算。报告中的向量是投影后的实际验算向量，严格检查器本身仍拒绝错误符号。`dual_objective_estimate` 可能不构成有效界，因此不代替 `best_bound`；例如非零残差指向无限变量界时，后者为空，数值残差仍可能在容差内。修正界也只按浮点算术计算，**没有精确算术或有向舍入认证，不能据此宣称获得严格剪枝证书**。

自主路径拒绝整数模型且不会回退 HiGHS。IPM 有显式固定变量代入实验；尚未实现一般不可行/无界证书、其他通用预求解、外部热启动或构基；不收敛时返回未完成或数值错误。

HiGHS 使用进程级线程调度器，其后端求解在同一进程内串行进入，避免不同线程配置冲突；单次求解可配置后端线程数。自主文件路径复用受保护的解析阶段，随后释放后端锁，进入不依赖 HiGHS 的数值内核。当前没有并行或集群扩展性结论。

## 回归入口与下一步

[第59轮齐次分母验收候选](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)已归档并撤回，实验入口为[v62-run](experiments/data/2026-10-12-halpern-ipm/v62-run.py)与[v62-freeze](experiments/data/2026-10-12-halpern-ipm/v62-freeze.py)。候选源码及模式以该轮冻结包为准；当前CLI没有新增scalar账本或homogeneous_extended模式。恢复后的来源与本轮起始状态逐字节相同，保留平台探测测试隔离修复；回归、具体身份和下一步取舍统一由实验正文维护。

构建脚本默认运行 CTest：[模型与验证](../tests/test_verification.cpp)、[原始/对偶检查](../tests/test_lp_optimality.cpp)、[自主内核](../tests/test_pdlp.cpp)、[等价变换与退化边界](../tests/test_pdlp_adversarial.cpp)、[HiGHS 集成](../tests/test_backend.cpp)、[CLI 与基准运行器](../tests/test_cli.py)、[构造最优点对照](../tests/test_pdlp_comparison.py)、[比较协议](../tests/test_compare_lp.py)。实际执行次数和覆盖范围以[实验索引](experiments/readme.md)中的记录为准。

当前先根据本机 CPU 的固定公开集结果改善收敛与算子效率，再评估 GPU 算子和集群通信。迭代中的稀疏乘与转置乘已集中到算子边界；将来使用 GPU 还需要设备驻留向量、投影和归约实现，不能只替换一次矩阵乘就声称完成移植。资源型号、互联和访问入口尚待获取，未在本机测试中推断集群能力。

方向级估计、图像残差与 LD/转回 double 候选的诊断证据见[第三十九轮](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)。该轮只运行冻结副本，未改变生产能力、默认值或公开成绩；同加权 L2 的观测不能替代现有三块无穷范数验收。

[第五十轮HSD诊断](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)继续采用冻结副本，构建、采集、离线分析和核验入口分别为[v53-build](experiments/data/2026-10-12-halpern-ipm/v53-build.py)、[v53-run](experiments/data/2026-10-12-halpern-ipm/v53-run.py)、[v53-offline](experiments/data/2026-10-12-halpern-ipm/v53-offline.py)、[v53-verify](experiments/data/2026-10-12-halpern-ipm/v53-verify.py)。诊断环境开关只存在于独立程序，生产不支持该采集能力；运行器拒绝覆盖冻结结果。当前点、保留候选、额外成本和未覆盖的assess预算中断边界由该轮正文统一说明，不改变生产选项或验收精度。

[第五十二轮阶段计时](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)使用冻结v54副本，入口为[v55-build](experiments/data/2026-10-12-halpern-ipm/v55-build.py)、[v55-run](experiments/data/2026-10-12-halpern-ipm/v55-run.py)和[v55-verify](experiments/data/2026-10-12-halpern-ipm/v55-verify.py)，插桩规则由[v55-instrumentation](experiments/data/2026-10-12-halpern-ipm/v55-instrumentation.py)维护。独立程序复用v47计时器，故仍以`SOLVER_V47_PROFILE=1`开启，标准错误仍以`V47_PROFILE `输出；这不是新增生产CLI。运行器先清理继承的`SOLVER_V`诊断环境，再显式设置本模式，拒绝覆盖冻结结果。记录包含/独占时间及正常/异常退出次数；CHOLMOD库入口与装配、拷贝、检查和恢复分别记账，嵌套包含时间不能直接相加。重复的阶段中位数不要求加总守恒，账本与数值轨迹的实际验证及下一候选由该轮正文统一维护。

[第60轮宽常数恢复阶段诊断](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)使用冻结v62独立副本，入口为[v63-build](experiments/data/2026-10-12-halpern-ipm/v63-build.py)、[v63-run](experiments/data/2026-10-12-halpern-ipm/v63-run.py)与[v63-analyze-hsd](experiments/data/2026-10-12-halpern-ipm/v63-analyze-hsd.py)。仅该诊断程序支持`SOLVER_V63_DIAGNOSTICS`；正式运行使用第二次冻结构建，拒绝覆盖已有证据。采集、预算中断及最后未提交候选的语义、实际验证和下一步取舍统一由本轮实验正文维护，当前生产接口、默认值与精度保持原状。

[第61轮宽候选乘积精度试验](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)的固定运行与冻结入口为[v64-run](experiments/data/2026-10-12-halpern-ipm/v64-run.py)、[v64-freeze](experiments/data/2026-10-12-halpern-ipm/v64-freeze.py)。候选沿用显式homogeneous_extended模式，完整源码以本轮冻结包为准；本轮起始95份来源另按原字节备份，不从文本记录重建混合行尾。构建、门槛、独立核验、恢复状态及后续取舍统一由该轮正文维护，不把解析测试通过当作默认生产能力已经升级。

[第62轮宽常数恢复加权试验](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)的运行与冻结入口为[v65-run](experiments/data/2026-10-12-halpern-ipm/v65-run.py)、[v65-freeze](experiments/data/2026-10-12-halpern-ipm/v65-freeze.py)。本轮沿用已有模式和工作计数字段；起始身份采用上一轮实际恢复状态，候选范围、加权账本、构建及取舍统一见实验正文，不从累计恢复次数推断一定执行过加权步骤。

[第63轮CHOLMOD矩形输入试验](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)的运行与冻结入口为[v66-run](experiments/data/2026-10-12-halpern-ipm/v66-run.py)、[v66-freeze](experiments/data/2026-10-12-halpern-ipm/v66-freeze.py)，实际选中来源见[接纳状态](experiments/data/2026-10-12-halpern-ipm/v66-accepted-state.json)。保留路径通过显式normal_cholmod_aat选择，沿用原单行消元要求与动态正则限制；已有normal_cholmod和默认值保持。实现范围、固定500条结果、独立核验及接纳依据统一由实验正文维护，不把扩集门槛通过等同于打平HiGHS或自动修改默认配置。

[第64轮固定空间诊断](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)的入口为[v67-build](experiments/data/2026-10-12-halpern-ipm/v67-build.py)、[v67-run](experiments/data/2026-10-12-halpern-ipm/v67-run.py)和[v67-analyze](experiments/data/2026-10-12-halpern-ipm/v67-analyze.py)。仅独立诊断程序支持SOLVER_V67_CAPTURE，生产无新增接口。工具拒绝覆盖已有证据；离线HiGHS调用不属于自主性能。本轮实际读取失败、固定原系数组装检查、独立核验及未实现事项统一由实验正文维护，生产保留状态和默认选择不变。

[第65轮基解恢复试验](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)的入口为[v68-build](experiments/data/2026-10-12-halpern-ipm/v68-build.py)与[v68-run](experiments/data/2026-10-12-halpern-ipm/v68-run.py)，独立程序以SOLVER_V68_BASIS_RECOVERY=1开启，生产无该接口。一次恢复仍须现有原模型验收；Eigen内部调用仅能前后检查预算，实际超限保留为失败。固定500条结果、侧账本、证据复核和不集成决定统一由实验正文维护；当前生产保持v66，不能把扩集门槛通过解释为生产接纳。

[第66轮左看消元选基](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)入口为[v69-build](experiments/data/2026-10-12-halpern-ipm/v69-build.py)与[v69-run](experiments/data/2026-10-12-halpern-ipm/v69-run.py)，独立程序以SOLVER_V69_BASIS_RECOVERY=1开启。选择循环有单独工作额度，侧账本区分已选主元与原模型验收；该额度不替代总时间限制。源码依据、22项自测、500条对照和未集成取舍统一由实验正文维护，生产接口不变。

[第67轮选择额度试验](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)入口为[v70-build](experiments/data/2026-10-12-halpern-ipm/v70-build.py)与[v70-run](experiments/data/2026-10-12-halpern-ipm/v70-run.py)，独立程序以SOLVER_V70_BASIS_RECOVERY=1开启。本轮只提高内部工作额度，原共同时间保持；实际恢复与终验拒绝、未扩完整集及下一诊断由实验正文维护。生产继续保留v66。

[第68轮基候选精确诊断](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)入口为[v71-build](experiments/data/2026-10-12-halpern-ipm/v71-build.py)、[v71-run](experiments/data/2026-10-12-halpern-ipm/v71-run.py)与[v71-analyze](experiments/data/2026-10-12-halpern-ipm/v71-analyze.py)。仅独立诊断程序支持SOLVER_V71_CAPTURE，并沿用原V70恢复开关；采集有容量与时间限制，分析采用捕获binary64的精确有理数。实际18条、采集开关兼容、基内外残差、原模型gap区别及结项的额外侧账本检查由本轮正文维护；未改变生产能力或正式成绩。

[第69轮同因子基精修](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)入口为[v72-build](experiments/data/2026-10-12-halpern-ipm/v72-build.py)与[v72-run](experiments/data/2026-10-12-halpern-ipm/v72-run.py)，独立程序以SOLVER_V72_BASIS_RECOVERY开启。新增两侧最多一次残差修正和独立账本，平台有效位数明确记录；原模型终验仍决定提交。实际29组自测、136条对照、未扩集和不集成决定由本轮实验维护，生产继续保持v66。

[第70轮有限两阶段换基](experiments/2026-10-12-Halpern与稀疏内点法迭代.md)入口为[v73-build](experiments/data/2026-10-12-halpern-ipm/v73-build.py)与[v73-run](experiments/data/2026-10-12-halpern-ipm/v73-run.py)，仅独立程序支持SOLVER_V73_BASIS_RECOVERY。初始构基和两阶段修复分别记账，最终仍须原模型终验。42组内核测试、500条对照、入口残差守卫失败及下一分辨实验统一由该轮正文维护，生产接口与保留状态不变。
