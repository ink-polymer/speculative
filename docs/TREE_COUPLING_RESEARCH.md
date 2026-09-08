# 树上块验证：面向论文主方法的研究转向

后续研究入口已转到 [单步块扩散树上块验证](DIFFUSION_TREE_BV.md)。本文保留
RM-BV 的历史推导与对照身份，不代表当前完整随机树构造或新颖性结论。

日期：2026-09-07。

状态：已实现可穷举的概率 oracle，以及 Root-Marginalized Block Verification
（RM-BV，内部工作名）的串行参考版、张量扫描/终止事件采样版和机制消融，
并已接入 `Engine.generate` 的真实 Target/DFlash 模型前向、树注意力和 KV 回收。
下面给出实数算术下的组合证明。
**本地 tiny Qwen 集成通过；尚无 GPU 速度结果，尚未确认论文新颖性。**
TM-TBV 留作工程基线，不再将其相同概率律的重新实现当作主算法贡献。

## 1. 改变的对象与研究边界

固定树 T，要求输出在给定 T 后仍与 Target 分布一致，且每次只能返回一个树前缀
加一个 correction token。此时 DDTree 已达到

    E[accepted | T] = sum_{v in T, v != root} p(v).

因此，仅重新排列固定树的采样顺序不能提高这个期望。
这一点不限制对随机候选树和输出做联合耦合：只要求将候选树的随机性积分掉以后，
完整输出仍是 Target 分布。普通 speculative sampling/BV 已利用过这一差别；
不能把这个观察本身称为新理论。

本轮考察的是：让多个首词候选共用一个随机后缀块，暂不决定首词，在精确边际化的
Target 后缀分布上进行 BV，再通过后验分布恢复首词。它改变候选树的随机分布和
验证决策的信息依赖，属于**推理解码算法/执行架构**，不改变 Target/DFlash 网络结构或权重。

## 2. RM-BV：具体构造

固定当前已提交上下文 c。令首词候选集合 S={a_1,...,a_K}，由采样后缀之前的信息
确定，例如 DFlash 第一个位置的 top-K，包含互不重复的 token。
S 不得根据本轮已抽到的后缀或其验证结果事后筛选。

从实际 proposal Q 抽一个长度 m 的后缀 z。使用 DFlash 时，

    Q(z_1:m) = product_j q_{j+1}(z_j | c).

这里保留实际抽样所使用的 q；不能把抽样后的 top-k 分数当成未经归一化的 proposal。
为每个 a 构建路径 (a,z_1,...,z_m)。这些路径共用上下文，但各自的祖先和 KV 不同。
相同后缀 token 不是可跨分支合并的 KV 节点。
一次正常祖先遮罩的 Target 前向得到：

    p(a | c), 全词表首词分布；
    p(x | c,a,z_1:j), 对每个 a in S、j=0,...,m 的全词表分布。

树包含 K(m+1) 个非根节点。对应原主实验 B=45、L=15 的候选实例是
K=3、m=14；不能额外暗中增加验证节点。

令 Z=sum_{a in S} p(a|c)。

1. 以概率 1-Z，从 S 之外按真实 Target 首词概率采样并返回一个 token。
   实现用显式屏蔽 S 后求和，不用易消去的 1-sum(S) 计算极小尾部。
2. 以概率 Z 进入块验证。首词仍未被选定，初始权重为
   alpha_0(a)=p(a|c)/Z。
3. 沿共同后缀构造精确的 Target 混合条件分布：

       M_j(x) = sum_a alpha_j(a) p(x | c,a,z_1:j)
       alpha_{j+1}(a) ∝ alpha_j(a) p(z_{j+1} | c,a,z_1:j).

   必须混合归一化后的概率，不能平均 logits 或 hidden states。
4. 调用已知的单链 BV，对 z、M、Q 联合验证，得到后缀
   u=(z_1,...,z_tau,b)，其中 b 为 correction/bonus token。
   这里是真正的 BV，不是把普通逐 token 验证重新命名为块验证。
5. 根据完整已发出后缀（**包括 b**）选择首词：

       Pr(a | u, a in S) ∝ alpha_tau(a) p(b | c,a,z_1:tau).

6. 返回 (a,u)，只保留 a 对应路径上已接受节点的 KV；b 作为待处理 token。

若 Z=0，仅执行外部首词分支；若 Z=1，不执行外部首词分支。遇到真实 Target
零概率后缀时，后续形式条件分布可任意补全，BV 不会提交零概率前缀；参考实现
对最后恢复首词时的零质量或数值下溢明确报错。

## 3. 无偏性的组合证明

省略 c，定义条件联合分布

    P_S(a,y) = p(a,y)/Z, a in S,
    M(y) = sum_a P_S(a,y).

上述 alpha 递推恰好给出 M 的自回归条件分布，仅需要树中已经得到的 Target 行。
BV 的可组合性结论是：如果它输出变长后缀 u，然后按 M 的条件分布补到固定长度
m+1，最终 y 的分布为 M。记 h(u) 为积分掉 proposal 和 BV 随机性后的返回概率，则

    sum_{u prefix of y} h(u) M(y_remaining | u) = M(y).

对于 P_S(a,y)>0，按后验恢复 a，再用原 Target 在 (a,u) 后继续生成，有

    Pr(output completed to (a,y))
      = sum_{u prefix of y} h(u) P_S(a|u) P_S(y_remaining|a,u)
      = P_S(a,y) sum_{u prefix of y} h(u)/M(u)
      = P_S(a,y).

再乘入口质量 Z，并加上 a 不在 S 时直接从 Target 抽首词的分支，得到完整的 p。
这同样说明 correction token 必须进入首词后验；忽略它不是有效消融，而是可能有偏。
每次返回都能用真实 Target 条件分布继续生成，因此可按 BV 的组合条件逐轮应用。

这是基于已知 BV 和 Bayes 后验的**本地组合推导**，不是对已有全部树验证方法的
最优性证明，也不是已获审稿认可的新定理。BV 的基础归属应明确引用
[Block Verification](https://arxiv.org/abs/2403.10444)。

数值边界：上述证明使用精确实数。参考实现使用 FP64，穷举测试检验误差；
不承诺浮点实现与精确实数逐比特相同。GPU GEMM/attention 的数值差异另需验证。

## 4. 已验证的正例与反例

二元词表、两个草稿位置、4 个非根节点、再加一个 bonus token。
两种树族都先保证首词 0、1 各有一个分支，且每个分支长度为 2。

- shared-second：以各 1/2 概率选 {00,10} 或 {01,11}。
- xor-second：以各 1/2 概率选 {00,11} 或 {01,10}。

随机交换路径索引后，两种树族的每条带索引路径都服从相同的均匀分布。
两者节点数、叶子数、深度完全相同；不是“多算几个节点”产生的提升。

所有数值均为**期望每轮输出 token 数，包含 correction/bonus，不是 tok/s**：

| Target 的第二词规律 | 最佳固定 4 节点树 + DDTree | shared-second 联合 oracle | xor-second 联合 oracle | 当前 RM-BV 参考实现 |
|---|---:|---:|---:|---:|
| 90% 复制第一词 | 2.9 | 3.0 | 2.6 | 3.0 |
| 90% 为 0，与第一词无关 | 2.9 | 2.6 | 3.0 | 2.6 |

对两种随机树都改用 DDTree 的给定树条件采样，结果均为 2.5。
“最佳固定树”是穷举全部合法 4 节点树得到的强对照，并不是声称 DDTree 的实际
Draft 驱动构树器知道这些 Target 概率。

解释：延后首词选择确实可以得到固定树机制无法取得的接受长度；但当前共享后缀
结构在另一个 Target 上输给最佳固定树。不存在本结果支持的普遍支配或 GPU 加速保证。

oracle 对完整字符串施加约束，不只核验首词或各位置边际。其变量为
f(T,u,x)=Pr(tree=T, return=u+x)，约束为

    f >= 0;
    sum_{u in T,x} f(T,u,x) = Q(T);
    sum_{T,u+x prefix of y} f(T,u,x) p(y_remaining | u+x) = p(y).

目标是最大化 sum f(T,u,x)(|u|+1)。额外要求每棵树分别匹配 Target 就恢复固定树
条件约束。该 LP 只是全信息上界工具，使用数值 SciPy/HiGHS，不能充当线上算法。
另有 Fraction 有理数证书精确验证正例及带拒绝回退的反例输出律。

### 4.1 多步分离构造：不是仅有两个位置的巧合

还可构造一个人工 Target：先从联合分布角度取 m 个独立均匀二元后缀位 z，
首词 a 以 90% 概率等于这些位的奇偶校验、以 10% 概率相反，最后 bonus 独立均匀。
这定义了一个合法、全支持的左到右自回归 Target：首词均匀，随后前 m-1 个后缀位
均匀，最后一个后缀位依据已知奇偶关系呈 0.9/0.1 分布。

取 S={0,1}、Q 为 IID 均匀后缀，则边际 Target M 恰好等于 Q，RM-BV 每次都
接受全部 m 个后缀，输出 m+2 个 token，使用 B=2(m+1) 个非根节点。
而任意固定树在深度 d<=m 的每个前缀概率为 2^(-d)，最后层每个前缀概率
至多 2^(-m)，所以

    E[committed_fixed] <= 1 + sum_{d=1}^m min(1,B/2^d) + min(1,B/2^m)
                       = O(log B).

因此，在这个人工分布族上存在接受长度的渐近分离，不只是浮点 LP 的偶然结果。
它不涉及对现代 LLM 权重的分布假设，不是实际 wall-clock 分离，也不声称这是
此前文献未曾使用的分离思想。

实际代码穷举到 m=1,2,3,4；分别得到 RM-BV 的 3、4、5、6 个 token/轮，
同预算最佳固定树为 2.9、3、3.25、3.5。第二词偏向常数的失败例仍保留，
不能用此人工族推断所有 Target 都会提高接受长度。

## 5. 为什么可能更快，以及为什么也可能失败

DFlash 的位置边际可以接近“首词尚未决定时”的 Target 后缀分布，而不接近某个
已经固定首词的条件分布。RM-BV 直接在前一种分布上验证，并让后缀信息帮助决定
首词。这是研究动机，不是关于真实 LLM 的已测结论。

额外概率处理为 O(K m V)，保存混合行需要 O(m V)，保存首词后验需要 O(K m)。
一次 Target 树前向不变，但树的形状变了。相同 B 不能自动保证相同 attention 延迟。
参考实现有按深度的 Python 循环、校验和同步，不能拿它的时间作为最终 GPU 架构结论。
后续可研究并行前缀权重与混合归约，但不能把融合 kernel 说成新的无偏性原理。

主要风险：首词集合覆盖质量 Z 低；后缀边际仍严重错配；固定 K 条长分支浪费节点；
全词表混合的访存和同步抵消接受增益。只对一次 Target forward 计时会漏掉这些成本。
最终需满足

    E[committed_RM] / E[committed_DD]
      > E[round_time_RM] / E[round_time_DD]

才有每轮均值意义上的吞吐收益；真实整请求吞吐仍以未 profile 的实测为准。

### 5.1 原创性/速度要求加严后的淘汰条件

用户要求的是“原创、足以支撑顶会主方法、比 DDTree 快”同时成立。这三项是
交付的验收目标，不能在实验之前标记为已达成，也不能用通过单元测试代替。

对当前候选可以立即得到一个保守上限。令 L=m+1，首词集合覆盖率为 Z，则

    E[committed_RM | c] <= (1-Z)*1 + Z*(L+1) = 1+LZ.

即使共享后缀永远全收，这个结构也越不过该上限。穷举真实验证器输出律的测试
现在同时检查这一点。它是门控结构的简单推论，不是拟主张的新定理。

在同一批固定开发上下文上，若有 RM-BV 每轮耗时的有效下界 t_lower(c)>0，
那么 sum_c(1+LZ(c))/sum_c t_lower(c) 是该开发状态分布上的宽松吞吐上界。
若连该上界都不超过同状态 DDTree 对照，就没有理由继续该配置。
这里的时间下界不能拿“另一个树形的 Target 耗时”随意代替；必须计入或合法下界
当前候选自己的 draft、Target、概率混合、采样与 cache 成本。通过该筛选也不代表
整请求加速，后续仍必须做闭环生成评测。

若同状态平均每轮计算成本相对 DDTree 增加比例为 delta，那么仅为收支相抵，
平均提交长度也至少要增加相同比例；若目标速度比为 s，还需要长度比超过
s*(1+delta)。这些是成本核算关系，不是提前预测会获得某个速度比。

## 6. 新颖性风险清单（有限检索，不是“保证不重复”）

- [GBV](https://arxiv.org/abs/2602.16961)：多路径 LP、选择路径后做 BV 已有。
  当前候选用共享后缀和延迟恢复首词，不是 IID 多条路径的贪心挑选；通用 LP 不能算贡献。
- [UniVer](https://arxiv.org/abs/2605.04543)：跨层概率分配与后序决策已有。
  需要检查其条件框架能否直接覆盖当前的跨分支相关 proposal；不能仅以决策顺序不同主张新颖。
- [Traversal Verification](https://arxiv.org/abs/2505.12398) 和
  [Layer Verification](https://chulheeyun.github.io/publication/cha2026layer/)：
  自底向上/后向决定已是相关方向。特别是后者，本轮新增为待精读和可化约性检查对象，
  尚不能排除它覆盖当前构造；不能以“最后选首词”这一描述单独主张新颖。
- [SpecTr-GBV](https://arxiv.org/abs/2604.25925)：多草稿块最优传输已有。
  其特定耦合假设、跨轮分布修改与本候选的 target-completion 条件需逐条比较；
  不把其最优性结论泛化为所有随机树的全局最优。
- [RSD](https://arxiv.org/abs/2402.14160)、
  [List-Level Coupling](https://arxiv.org/abs/2506.05632)：无放回、多样化、共同随机性已有。
  “候选相关”本身不构成新颖性。
- [DBLast](https://arxiv.org/abs/2608.05448)：学习草稿的潜变量混合以及正确 proposal 后验已有。
  当前候选边际化的是固定 Target 的显式首词分支，最后恢复真实首词，且不训练新的草稿混合头。
- [DARTree](https://arxiv.org/abs/2608.13524)、
  [JetFlow](https://arxiv.org/abs/2606.18394)、
  [SpecBlock](https://arxiv.org/abs/2605.07243)：分支依赖、并行/块式草稿网络已有。
  不把“给 DFlash 加分支条件 head”当成尚未有人做的方向。
- [LiLiCorr](https://arxiv.org/abs/2608.20530)：利用轻量模块修正平行草稿相关性已有。
  当前不训练此类修正网络，但正式 related work 仍需比较。
- [ASSD](https://arxiv.org/abs/2504.20456)：改变抽样顺序并保持正确联合分布已有。
  本候选用普通左到右 Target 的有限树概率实现局部边际化，不假设 Target 已是 any-subset 模型。
- [CaDDTree](https://arxiv.org/abs/2606.01813)：联合结构/预算的吞吐优化已有。
  预算调整本身不是当前候选的贡献。

尚待完成：针对先前方法的可化约性检查、更多关键词/引文链检索、独立专家审阅。
未找到同名方法不意味着机制不存在。当前不能写“首个”“已确认全新”“顶会级已完成”。

本轮补充核查记录：Layer Verification 的作者页面及其索引片段描述了节点打分、
局部流与后向采样，说明它是重要近邻；这些信息尚不足以证明它覆盖或不覆盖 RM-BV。
作者所链 OpenReview 页面（XrqcUo6B6L）及已索引 PDF 的直接访问均被浏览器验证
拦截，未完成 PDF 公式/图示的完整核对，也未绕过验证。故此项明确保留为未关闭的
新颖性风险，不把“全文暂不可读”当作原创性证据。可以用用户提供的全文补齐核查。

同样，已有 GH200 上旧 BRBV/稀疏验证变体的时间记录不能替代当前 RM-BV 的
H200 测量。当前没有支持 RM-BV 快于 DDTree 的真实 GPU 结果。

## 7. 正式实验扩展：不改变旧数据，不把参考实现冒充成完整系统

保留原 7 组数据：GSM8K、MATH500、AIME25、HumanEval、MBPP、LiveCodeBench、MT-Bench；
复用原固定样本、prompt、模型 revision、seed、2K 上限。没有远程冻结数据时不重新下载
最新版本冒充旧数据。原 TM-TBV 的正式实验脚本仍是该旧方法的协议，**不会自动验证 RM-BV**。

已接入生成引擎、待 GPU 测量的同树分布、同预算机制消融：

1. shared-suffix tree + DDTree：隔离树结构与耦合验证的贡献。
2. 同一树先按 Target 选择首词，再对所选分支用标准 BV：隔离延迟首词决定的贡献。
3. 在相同精确边际 M 上用逐 token 验证，再做同样的首词后验恢复：隔离 BV 的贡献。
4. 完整 RM-BV。保持相同 proposal、B、L、Target 前向和精度，计入每种验证器全部成本。

另保留串行 RM-BV 参考版，隔离张量扫描/终止事件实现的作用。
这里“同树分布”不意味着不同验证器的整请求会走到相同上下文或抽到完全相同的树；
严格同一状态、同一棵树的耗时/接受长度比较需要单独捕获与重放，不能由共享 seed 冒充。

独立后缀不能直接传给当前共享后缀验证器作为开关消融，那会破坏输入概率假设；
独立多路径对照应使用其自身正确的 GBV/其他验证器。去掉 correction 后验也不是有效消融。

主对照至少包含 AR、官方一致的 DDTree、DFlash，并补最接近的可运行树块验证方法。
单一 Qwen3-4B 结果只支持该设置；推广性主张需进一步在 8B/其他模型设置验证。
对额外训练过的草稿方法，需分开报告不同权重下的整系统比较，不混成固定模型的架构消融。

评估顺序：穷举完整输出律和支持约束 → 多轮/EOS/截断/KV 集成正确性 →
独立开发集的真实状态诊断与端到端 pilot → 冻结方法/代码/数据 → 原 7 数据集正式评测。
保留原多 seed、多次重复、配对 bootstrap、各数据集结果、原始输出和失败记录。
速度看端到端解码，质量保持原数学/代码评测；MT-Bench 无正式 judge 时不声称质量已完成。
失败设置和负结果必须保留，不依据测试集结果筛选配置或报告子集。

登录后先核验实际 GPU 型号，再提交 Slurm 作业；H200 与 GH200 不混写。
目前没有由本文档产生的新远程实验或已完成的 H200 结果。

## 8. 当前可复现内容

- `src/gbv_experiments/tree_coupling_oracle.py`：有限树混合 LP、正反例及强固定树对照。
- `src/gbv_experiments/root_marginalized_bv.py`：参考算法、张量化实现、共享后缀 proposal、有效消融。
- `src/gbv_experiments/engine.py`：一次 Target 树前向、后验分支恢复、对应 KV/features 回收。
- `configs/root_marginal_bv_qwen3_4b.json`：10 个方法的固定设置，不替换旧 TM 正式协议。
- 两个对应测试文件：有理数证书；穷举真实采样调用；部分首词支持、零概率、
  多深度、自回归依赖、correction 后验和无效输入。

运行：

```bash
.artifacts/gbv-test-venv/bin/python -m pytest -o addopts='' \
  tests/gbv_paper/test_tree_coupling_oracle.py \
  tests/gbv_paper/test_root_marginalized_bv.py -q
PYTHONPATH=src .artifacts/gbv-test-venv/bin/python -m gbv_experiments.tree_coupling_oracle
```

此前仅参考算法的 385 项记录保留在 `.artifacts/tree-coupling-research-local-tests.xml`。
本轮优化和模型集成的全目录回归记录在
`.artifacts/rm-bv-tensorized-integration-tests.xml`；不要混用两份证据。
这不是 GPU 正式 checkpoint 或正式数据集实验通过的记录。

真正的下一道门槛是可高效实现、具有明确非重复机制的算法，在完整公平评测下快于
DDTree 和 DFlash。论文目标可以设高，不能提前保证性能、无条件新颖性或录用。

## 9. 本轮实际落地的优化

### 9.1 张量扫描与终止事件采样

对根分支 a 和后缀深度 j，用 log 概率前缀和一次性构造

    ell[a,j] = log p(a) + sum_{r<j} log p(z_r | a,z_<r)
    alpha[:,j] = softmax_a(ell[:,j]).

全零目标前缀的条件行任意补全，但它的 BV 前缀权重为零，不能从那里发出输出。
概率混合使用 `einsum`；没有平均 logits，也没有额外 Target 网络前向。
树节点按分支连续排列，因此 `all_p[1:].view(K,L,V)` 可避免再复制 K×L×V 概率张量。

将 BV 的前缀权重递推写成 scan：

    s_0=0; s_i=sum_{j<i} log(M_j[z_j]/Q_j[z_j])
    w_i=exp(s_i-max_{0<=j<=i}s_j).

令 R_i=(w_i M_i-Q_i)_+，最后一行 Q_m=0；将逐行 BV 的成功率记为 h_i。
BV 返回的是最深的成功行，因而

    Pr(tau=i | z)=h_i * product_{j>i}(1-h_j).

用反向累乘直接计算该分布，再把首词集合外的事件一起合并抽样。随后只抽选中行的
correction 和首词后验。**验证核心固定三次分类抽样**，并且只在最后显式 `.tolist()`
取回选定结果。串行参考版进入集合内分支时为 m+3 次，即 L=15 时为 17 次。
这个计数不包括 draft 后缀抽样、初始 anchor、树构建和模型前向。
PyTorch 内部仍可能同步；没有 GPU profiler 证据时不声称整个流水线只有一次同步。

稠密混合/残差仍需 O(KLV)/O(LV) 工作；本轮没有实现融合 CUDA/Triton kernel，
也没有把算子数量减少等同于实测加速。终止事件重写、扫描和减少拷贝本身只作为实现优化，
不单独主张原创性。论文算法候选仍是第 2 节的目标分支边际化与共享 proposal 耦合。

后续反向审计已找到同树下 RM-BV 的多步期望提交长度低于 early-root BV 的
严格正概率反例，见 [三轮核查](RM_BV_THREE_PASS_RECHECK.md)。此前复制/奇偶族
正例不能推广成多步无条件优势；通用框架级原创性也尚未通过。

### 9.2 有效且能运行的对照

| 配置名 | 实现 | 隔离对象 |
| --- | --- | --- |
| rm_full | 延迟选根 + 边际 BV + 三次抽样 | 完整候选 |
| rm_serial_ref | 同一 RM-BV 返回块概率律，逐深度参考实现 | 扫描和终止事件实现 |
| rm_token | 同一边际 Target，精确 token rejection，再恢复根 | 块验证是否贡献收益 |
| rm_early_root | 先按 Target 选根，再对该分支 BV；也用终止事件实现 | 延迟选择根分支 |
| rm_shared_ddtree | 同一共享后缀 proposal，用 DDTree 验证 | 候选结构与验证规则 |

其余主对照为 `target_t1`、`dflash_match`、`ddtree`，加单链 `dflash_bv` 与 `gbv`。
所有 RM 变体 K=3、L=15、45 个非 anchor 节点；超预算或 FP32 设置直接报错。
原七数据、模型 revision、样本选择 seed、生成 seed、2K 上限、温度、权重精度不变。
单次完整遍历的计划为 23,580 条 question/conversation 记录、25,980 次生成。
这是计划数量，不是完成数量，也不包含独立 timing repeats。

### 9.3 登录后的最短验证路径

复用远程原数据目录；先核对实际 GPU 型号及数据 manifest。以下命令仅准备入口，
本地没有调用 CUDA 或重新拉取数据；`<原数据目录>`、`<新输出目录>` 需要用实际路径替换。

```bash
PYTHONPATH=src python -m gbv_experiments check-model \
  --config configs/root_marginal_bv_qwen3_4b.json \
  --device cuda:0 --output <新输出目录>/model-check.json
PYTHONPATH=src python -m gbv_experiments run \
  --config configs/root_marginal_bv_qwen3_4b.json \
  --data-dir <原数据目录> --device cuda:0 --smoke \
  --output <新输出目录>/rm-smoke
```

`check-model` 包括原代码评分环境检查，默认要求 Docker；若集群不提供该隔离环境，
需先安排独立的代码评分环境，不应静默把未知生成代码改成宿主进程执行。
smoke 只检查连通性和集成，不能用于报告性能或在原测试集上选择结构。
随后必须用独立开发上下文做实际 Qwen checkpoint 的同状态重放和闭环 pilot。
GPU 型号核验、重放计时、三次 timing repeats 的跨重复统计和正式冻结仍是后续门槛；
现有通用 `run/score/report` 可执行完整配置，但不能把它自动称为 RM 的全部正式研究证据。
