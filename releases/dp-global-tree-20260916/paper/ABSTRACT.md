# 论文摘要草稿

建议标题：**面向并发块扩散推测解码的全局异构树预算调度**

## 中文

块扩散推测解码能够并行生成候选token，但在多请求并发下，统一使用大树会放大Target验证开销，单请求接受长度的提升不一定转化为系统吞吐收益。本文研究请求之间共享验证行预算的异构树调度问题。在冻结的块扩散Draft和DDTree构树机制上，我们根据Draft路径概率估计候选树收益，结合硬件校准的行数代价及因果请求权重，将单轮树档位分配建模为多选择背包问题。动态规划计算每个可达总行数下的最优代理收益，再选择收益与代价比最大的分配，实现请求间验证资源的自适应共享。我们证明了给定入场集合和候选集合内的代理目标最优性、相对可行固定分配的弱支配性、候选集合扩展的收益单调性，以及模型误差下的决策稳定性界。H20上覆盖两种Target模型、两种温度、三种预算和五种并发的60配置诊断实验中，温度1的30配置相对共同框架DDTree与DFlash的吞吐倍率几何均值分别为1.437×与1.226×，其中分别在26与30个配置胜出。独立审计验证了请求隔离及有限状态序列规律，同时发现BF16实现与规定AR参考之间存在数值差异。因此本文区分理想算法的条件性分布保持与现实实现的数值认证，不将当前结果解释为严格无损或任务质量已认证不变。

关键词：推测解码；块扩散；并发调度；异构树；共享验证预算；动态规划。

## English

Block-diffusion speculative decoding generates candidate tokens in parallel, but uniformly verifying large draft trees across concurrent requests can inflate target-model work. We study heterogeneous tree allocation under a shared verification-row budget. Building on frozen block-diffusion drafting and DDTree construction, we estimate candidate-tree utility from draft path probabilities and combine it with a calibrated row-cost model and causal request weights. We formulate each admitted verification wave as a multiple-choice knapsack problem. Dynamic programming computes the maximum surrogate reward for every reachable row count and selects the allocation with the highest reward-to-cost ratio. We establish surrogate optimality within a fixed admitted cohort and gated candidate set, weak dominance over feasible fixed allocations, monotonicity under candidate-set expansion, and decision-stability bounds under model error. A diagnostic H20 evaluation spans two target models, two temperatures, three budgets, and five concurrency levels. Across the 30 temperature-1 configurations, geometric-mean throughput ratios over common-framework DDTree and DFlash are 1.437× and 1.226×, with improvements in 26 and 30 configurations, respectively. Independent audits validate request isolation and finite-state sequence-law properties, but detect numerical differences between the BF16 implementation and the specified autoregressive reference. We therefore distinguish conditional distribution preservation in the ideal algorithm from numerical certification of its implementation; task-quality preservation and production-serving gains remain unverified.

## 使用边界

这些是本工作数学模型的性质，不是首次提出背包DP或分布保持理论的声明。正式投稿前需要直接相关方法对比、同路径调度消融、动态到达serving、独立任务质量评估及统计不确定性。统计见EVIDENCE_SUMMARY.json。
