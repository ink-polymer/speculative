# AdaptiveTree、DDTree 与 DFlash：T=0 / T=1 正式实验

> 本文档对应新服务器全矩阵分支。此前基于 `16c0e91` 启动过的任务不含树状块，
> 不得用本分支去 `resume` 它。树状块全矩阵必须在新服务器使用全新输出目录启动。

## 新服务器的实验边界

本套件只运行两组不能混合统计的正式实验：

- T=0：Qwen3-4B 与 Qwen3-8B 上的修正版 `adaptive` 主方法、冻结的 AdaptiveTree 消融、Target、DFlash，以及七个固定预算 DDTree。
- T=1：同一组 Qwen3-4B 与 Qwen3-8B revision 上的 Target、DFlash、DDTree 和 `tree_block_verification`；DDTree 与候选固定 L=15、B=45、FP64 概率计算。

树状块验证固定使用 `ddtree_fused_scan`，与 DDTree 共用同一棵 `probability_tree`
及全部 Target 概率行，仅替换祖先采样验证器。T=0 与 T=1 使用不同采样律，禁止
合并加速比或置信区间。30B 不属于本轮正式矩阵。
每个真实模型的 GPU preflight 还会记录 T=1 同树 witness；正式报告必须
重验两种方法的 parents、tree tokens 和 Target FP64 概率张量哈希全部相同。

若先单独执行 H20 上的 4B T=1，该任务始终只是
**“Qwen3-4B / T=1 / H20 登记子矩阵”**（9,792 条结果记录、10,752 个实际生成轮次）。
它不包含 8B 或 T=0，所以即使子矩阵所有门禁通过，也必须保持
`formal_complete=false`、`whole_formal_matrix_complete=false` 和论文结论禁止状态。
全套正式完成只指 4B/8B、T=0/T=1 的 65,280-call 矩阵和全部完整性/公平性门禁通过。

## 公平性与失败即停止门禁

所有模型固定 40 位 Hugging Face revision、BF16、关闭 thinking 和 TF32。T=1 四个方法共享 Target=SDPA、Draft=SDPA、数据、三个种子、生成上限和评分器。DFlash 在 T=1 仍使用官方贪心 Draft；DDTree 与树块候选使用同一个 T=1 Draft 概率树。

方法执行顺序按数据集分别从 ordinal 0 开始，并跨 seeds 连续循环。报告会逐数据集检查每个方法落在每个计时位置的次数，最大差必须不超过 1；仅在全局看似均衡、但某个数据集有位置偏差的结果会被拒绝。

`doctor` 会拒绝非逻辑 `cuda:0`、GPU UUID 漂移或已有其他计算进程占用的机器，并实际
执行所选代码评分沙箱。Docker 后端绑定镜像 ID；process 后端保留虚拟环境调用路径，
并另行绑定真实二进制 SHA-256、`pyvenv.cfg` 与子解释器 Python/NumPy/SymPy 身份；root
编排器必须将参赛代码降权到非 root 身份。

`doctor` 还会在加载正式模型前运行两个确定性的有限枚举分布律测试：

- DFlash `matching_verify` 的完整输出律等于 Target 自回归律；
- DDTree batched ancestral verifier 的输出律等于逐节点 Target ancestral sampling。

此外还会用真实微型 Qwen3 前向检查 T=1 DFlash 的 Draft 确实保持 greedy
argmax，而不是误用 Target 的采样温度。随后每个正式 4B/8B checkpoint 都会分别
执行 DDTree 与树块候选，保存首棵树的父节点、树 token、FP64 Target 概率张量哈希
和形状；同时在各自验证器第一次成功返回后记录实际 callable、源码哈希以及该次
输入/输出哈希，其中输入还绑定该次调用前的 generator 状态。完整张量必须逐元素
相同、调用路由必须分别命中官方 DDTree 与 fused-scan 实现，运行时 witness 才通过。
observer 只用于 preflight，不进入正式计时路径。

归档 r2 的旧 observer 在 verifier 调用之前就触发；它可以保留当时准备的同树输入，
却不能独立证明所记录的 fused callable 实际被调用并成功返回。新正式 witness 必须在
真实 verifier 成功返回后才发出事件，并同时绑定实际 callable 身份、调用前 RNG 状态及
输出哈希。该新证据不得从 r2 归档倒推或回填。

测试 node ID、测试源码 SHA-256 和退出状态写入 `server_doctor.json`。随后在任何正式
计时之前，套件先完成 T=1 的真实数据/答案审计及两个真实模型预检，再用两个真实模型对
T=0 官方路径做 GPU smoke，覆盖双 Target 后端、FA2 Draft、C++ compaction 和完整方法注册表。
最终统一报告会重新核验这些证据；缺失、失败、测试名、来源或源码哈希不一致都会阻断报告。

T=1 报告还会核验 summary 的 run ID、bootstrap 次数、`performance_only=false`、每个 `(dataset, variant)` 恰好一行，以及除 MT-Bench 外每个样本都有客观质量评分。Markdown 主表同时列出质量、平均生成长度和每次 Target 验证接受的 token 数。DFlash 的提议分母是路径 token，DDTree 的提议分母是整棵树节点，因此不把二者的“接受/提议比”作为横向比较指标。
MT-Bench 的质量列显示 `--`，表示外部 judge 结果单独报告，并非漏评分。

## 新服务器运行

旧服务器结果不能导入。每张新 GPU 使用一个从未存在过的输出目录；环境、代码或配置改变后也必须换目录。

```bash
export INTEGRATED_PYTHON=/root/autodl-tmp/envs/speculative/bin/python
export INTEGRATED_RUN_DIR=/root/autodl-tmp/outputs/adaptivetree-t0-t1-tree-block-future-001
export ADAPTIVE_DATA_DIR=/root/autodl-tmp/data/adaptive-t0
export SAMPLING_DATA_DIR=/root/autodl-tmp/data/sampling-t1
export CODE_BACKEND=process
export GBV_PROCESS_PYTHON=/opt/gbv-code-eval/bin/python

bash scripts/run_integrated_fresh_server.sh plan
bash scripts/run_integrated_fresh_server.sh audit
bash scripts/run_integrated_fresh_server.sh doctor
bash scripts/run_integrated_fresh_server.sh start
```

`doctor` 必须成功后才能 `start`。`start` 使用后台进程，SSH 断开不影响运行；同一 contract 因断电或进程退出时使用 `resume`，不要重新 `start`：

```bash
bash scripts/run_integrated_fresh_server.sh status
bash scripts/run_integrated_fresh_server.sh resume
```

启动器对同一 `INTEGRATED_RUN_DIR` 持有完整 worker 生命周期锁；并发执行 `start` 或
`resume` 只允许一个进程进入，避免两个任务同时写结果或争抢 GPU。

## 主要产物

- `server_doctor.json`：冻结环境、GPU、评分沙箱、DDTree C++ 扩展及两项精确分布律证据。
- `formal_matrix_audit.json`：本次 worker 实际绑定的 canonical T=0/T=1 矩阵及其哈希。
- `fairness_audit.json`：revision、配置、源码和结论边界审计。
- `sampling_data_audit.json` / `sampling_gold_audit.json`：T=1 数据与客观评分答案审计。
- `adaptive_gpu_preflight.json`：两个真实模型的 T=0 双后端、全方法 GPU smoke 证据。
- `adaptive/<model>/tables.json`：T=0 全样本、输出差异及消融结果。
- `sampling_t1/<model>/gpu_preflight.json`：真实 checkpoint 的结构检查及 DDTree/树块同树运行时 witness。
- `sampling_t1/<model>/report/summary.json`：T=1 完整质量与性能统计。
- `report/integrated_results.md`：通过全部门禁后的 T=0/T=1 并列表格。
- `report/integrated_results.json`：机器可读结果，固定记录 `tree_block_verification=registered_t1_same_tree_fused_scan` 和 `cross_protocol_speedup_pooling_allowed=false`。
- `report/tree_block_t1_pairwise_vs_ddtree_and_dflash.csv`：从完整原始记录重算的 source 聚类配对区间。

汇总时不会信任已有表格：T=0 从所有原始 `.pt` 与完成标记重新建表，T=1 从
`results.jsonl` 和 `scores.jsonl` 重算覆盖与统计；重算结果必须与落盘表逐字段相同。
历史 pilot 只能用于工程资格判断，不会导入新正式计时。完整矩阵和
所有完整性/公平性门禁通过前，禁止把树块 pilot 数值当作正式结果；声称
优于某个基线时，该配对 source 聚类 95% CI 下界必须大于 1。
T=0 只有一次完整确定性 pass，只给描述性点估计、不提供置信区间；T=1 才报告冻结的
配对聚簇 bootstrap 区间。

只读计划、审计和重建报告分别使用：

```bash
PYTHONPATH=src python -m gbv_experiments plan-integrated-suite \
  --suite configs/adaptive_tree_block_suite.json
PYTHONPATH=src python -m gbv_experiments audit-integrated-suite \
  --suite configs/adaptive_tree_block_suite.json \
  --output outputs/integrated-fairness-audit.json
PYTHONPATH=src python -m gbv_experiments report-integrated-suite \
  --suite configs/adaptive_tree_block_suite.json \
  --run-dir "$INTEGRATED_RUN_DIR" \
  --output "$INTEGRATED_RUN_DIR/report"
```
