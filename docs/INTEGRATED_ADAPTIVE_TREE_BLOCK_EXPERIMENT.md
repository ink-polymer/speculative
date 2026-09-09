# AdaptiveTree、DDTree、DFlash 与新块解码的公平整合实验

## 结论边界

本套件统一启动、审计和汇总两类实验，但不把它们伪装成同一个采样协议：

- T=0：修改后的 AdaptiveTree、DFlash、七个固定预算 DDTree，以及两项成本归因/扩展预算诊断。
- T=0.3/0.6/1.0：Target、DFlash、DDTree 和 `ddtree_lazy_projection`。

T=0 与 T>0 的数据矩阵、随机性和 Draft 注意力后端不同，因此禁止把两类实验的加速比求平均或合并置信区间。统一报告只做并列展示。

## 公平性硬门禁

### T=0 AdaptiveTree

- 4B/8B Target 与 Draft 使用和块实验相同的固定 40 位 revision。
- Draft 固定 FA2；公平主表中的 Target、DFlash、DDTree、AdaptiveTree 全部取 Target=SDPA 的同后端结果。
- 上游“各方法选择最佳后端”的表仍会生成，但只标为辅助表，不用于架构优劣结论。
- 每个回答对方法顺序做确定性的循环轮换；完整循环中每个方法占据每个计时位置的次数相同，尾部差最多 1。
- 使用 `strict` 逐 token 门禁；任何方法与 Target-only 输出不一致即停止，不写成功结论。
- 修正后的 `cost_attributed_no_exploration` 与扩展到 B=256 的诊断同时运行，B=128 是 B=256 的同进程对照。

### T>0 块解码

- 4B/8B 均固定 BF16、Target=SDPA、Draft=SDPA、关闭 TF32 和 thinking。
- 每个温度都有成对的 Target、DFlash、DDTree、`ddtree_lazy_projection`，共享 L=15、B=45、FP64 概率计算和同一批样本/种子。
- DDTree 与 lazy projection 除方法名和实现入口外，所有配置字段必须完全一致。
- 12 个方法/温度组合采用固定种子的均衡轮换；每条结果记录实际执行位置，正式报告再次检查每题是否恰好覆盖全部位置。
- GPU 预检先在 T=0 检查真实 checkpoint 的贪心输出、树掩码和 KV 压缩；不通过则禁止正式计时。
- T>0 的质量使用相同任务评分器；速度使用逐题配对 Target 和以 `source_id` 为簇的 bootstrap 置信区间。

这些门禁能保证代码和协议层面的可比性，但不能把有限样本或单一 GPU 型号变成普适结论。正式表仍需报告 GPU、CUDA、库版本、置信区间和完整性状态。

## 使用方式

### 新服务器必须全量重跑

旧服务器的性能结果只能作为历史记录，不能复制进本套件。新服务器必须指定一个从未存在过的 `INTEGRATED_RUN_DIR`；GPU UUID、CUDA 环境、源码、配置和数据 revision 会写进新 contract。若把旧输出目录带到另一张卡上，环境检查会拒绝继续。

新服务器建议使用同一个 Python 3.11 虚拟环境，先按服务器驱动安装匹配的 CUDA PyTorch 与 `flash-attn`，再安装两套实验的共同依赖和代码评分镜像：

```bash
python3.11 -m venv .venv-integrated
source .venv-integrated/bin/activate

# H20/CUDA 12.8 的冻结环境：与官方 FlashAttention 2.8.3 wheel 匹配。
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gbv-paper.txt
python -m pip install -r adaptivetree_paper/requirements-paper.txt
# 安装官方 v2.8.3 的 cu12/torch2.8/cxx11abiTRUE/cp311 x86_64 wheel；
# 正式安装前核对 SHA-256：
# 3d41b2fc55753faa7f45d6568ea73a96b96afb48b82994ab9b49bcbcb6c87588

docker build -t gbv-code-eval:py311 experiments/gbv_paper
```

若租用的 GPU 容器没有 Docker 且内核禁止嵌套 user/network namespace，可显式设置
`CODE_BACKEND=process`。Linux root 作业的 process worker 会在编译、执行模型代码前不可逆地
降到 uid/gid 65534，并启用 `no_new_privs`、资源限制和清理后的环境变量。它不等价于禁网容器，
只应在无密钥、专用于本实验的实例中使用；评分清单会记录实际后端，不得把不同后端的评分混合。

先执行环境与协议检查，再用新目录后台启动：

```bash
export INTEGRATED_PYTHON="$PWD/.venv-integrated/bin/python"
export INTEGRATED_RUN_DIR="$PWD/outputs/integrated-new-server-001"

bash scripts/run_integrated_fresh_server.sh plan
bash scripts/run_integrated_fresh_server.sh audit
bash scripts/run_integrated_fresh_server.sh doctor
bash scripts/run_integrated_fresh_server.sh start
```

`start` 通过 `nohup` 启动，SSH 断开不会终止主进程。查看状态或断电/作业中止后从同一个新目录续跑：

```bash
bash scripts/run_integrated_fresh_server.sh status
bash scripts/run_integrated_fresh_server.sh resume
```

`start` 只接受完全不存在的输出/log/PID，防止误混旧实验；`resume` 只补当前 contract 缺失的记录。更换 GPU、源码或配置后必须改用另一个全新目录并重新 `start`。

本地只做计划和只读审计，不加载模型：

```bash
PYTHONPATH=src python -m gbv_experiments plan-integrated-suite \
  --suite configs/adaptive_tree_block_suite.json

PYTHONPATH=src python -m gbv_experiments audit-integrated-suite \
  --suite configs/adaptive_tree_block_suite.json \
  --output outputs/integrated-fairness-audit.json
```

在单张 NVIDIA GPU 上运行完整实验；输出目录可续跑，但配置、源码、数据或环境变化时必须换目录：

```bash
PYTHONPATH=src python -m gbv_experiments run-integrated-suite \
  --suite configs/adaptive_tree_block_suite.json \
  --adaptive-data-dir adaptivetree_paper/datasets/ddtree_official_t0 \
  --block-data-dir datasets/gbv_paper_ddtree_counts \
  --output outputs/integrated-adaptive-block \
  --device cuda:0
```

只重建统一表格：

```bash
PYTHONPATH=src python -m gbv_experiments report-integrated-suite \
  --suite configs/adaptive_tree_block_suite.json \
  --run-dir outputs/integrated-adaptive-block \
  --output outputs/integrated-adaptive-block/report
```

主要产物：

- `fairness_audit.json`：运行前的模型、后端、方法参数与源码审计。
- `adaptive/*_diagnostic/tables_controlled_sdpa.csv`：T=0 同后端主表。
- `adaptive/*_diagnostic/tables.csv`：T=0 最佳后端辅助表。
- `block/*/report/summary.csv`：T>0 各温度逐数据集表。
- `report/integrated_results.md`：通过全部门禁后的统一并列表格。
- `report/integrated_results.json`：机器可读结果；明确保存 `cross_protocol_speedup_pooling_allowed=false`。
