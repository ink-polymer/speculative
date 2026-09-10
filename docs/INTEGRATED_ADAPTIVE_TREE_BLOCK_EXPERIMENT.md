# AdaptiveTree、DDTree 与 DFlash：T=0 / T=1 正式实验

## 本服务器的实验边界

本套件只运行两组不能混合统计的正式实验：

- T=0：Qwen3-4B 与 Qwen3-8B 上的动态 `adaptive_b128` 主方法、以它为锚点的 AdaptiveTree 消融、Target、DFlash，以及七个固定预算 DDTree。
- T=1：同一组 Qwen3-4B 与 Qwen3-8B revision 上的 Target、DFlash、DDTree；DDTree 固定 L=15、B=45、FP64 概率计算。

树状块验证在本服务器明确为 `deferred_not_run`，不会启动，也不会出现在结果表中。T=0 与 T=1 使用不同采样律，禁止合并加速比或置信区间。30B 不属于本轮正式矩阵。

## 公平性与失败即停止门禁

所有模型固定 40 位 Hugging Face revision、BF16、关闭 thinking 和 TF32。T=1 三个方法共享 Target=SDPA、Draft=SDPA、数据、三个种子、生成上限和评分器。DFlash 在 T=1 仍使用官方贪心 Draft；DDTree 使用 T=1 Draft 分布。

方法执行顺序按数据集分别从 ordinal 0 开始，并跨 seeds 连续循环。报告会逐数据集检查每个方法落在每个计时位置的次数，最大差必须不超过 1；仅在全局看似均衡、但某个数据集有位置偏差的结果会被拒绝。

`doctor` 会拒绝非逻辑 `cuda:0`、GPU UUID 漂移或已有其他计算进程占用的机器，并实际
执行所选代码评分沙箱。Docker 后端绑定镜像 ID；process 后端保留虚拟环境调用路径，
并另行绑定真实二进制 SHA-256、`pyvenv.cfg` 与子解释器 Python/NumPy/SymPy 身份；root
编排器必须将参赛代码降权到非 root 身份。

`doctor` 还会在加载正式模型前运行两个确定性的有限枚举分布律测试：

- DFlash `matching_verify` 的完整输出律等于 Target 自回归律；
- DDTree batched ancestral verifier 的输出律等于逐节点 Target ancestral sampling。

此外还会用真实微型 Qwen3 前向检查 T=1 DFlash 的 Draft 确实保持 greedy
argmax，而不是误用 Target 的采样温度。

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
export INTEGRATED_RUN_DIR=/root/autodl-tmp/outputs/adaptive-b128-t0-t1-h20-001
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
- `sampling_t1/<model>/report/summary.json`：T=1 完整质量与性能统计。
- `report/integrated_results.md`：通过全部门禁后的 T=0/T=1 并列表格。
- `report/integrated_results.json`：机器可读结果，固定记录 `tree_block_verification=deferred_not_run` 和 `cross_protocol_speedup_pooling_allowed=false`。

汇总时不会信任已有表格：T=0 从所有原始 `.pt` 与完成标记重新建表，T=1 从
`results.jsonl` 和 `scores.jsonl` 重算覆盖与统计；重算结果必须与落盘表逐字段相同。
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
