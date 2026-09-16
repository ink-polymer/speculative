# 审计复测：本地测试记录

命令（adaptive checkout，`PYTHONPATH=src`）：

```sh
python3 -m pytest -q -o addopts='' \
  tests/gbv_paper/test_audited_global_tree_matrix.py \
  tests/gbv_paper/test_continuous_tree_block_decode.py \
  tests/gbv_paper/test_terminal_mass.py \
  tests/test_ddtree_builder.py
```

结果：146 passed，13 skipped，27.09 秒。13 个跳过项需要 CUDA，本地是 macOS。
这不是全仓库测试通过的声明。此前更大测试集合的 transformers/huggingface-hub
版本冲突没有在本轮通过修改环境来规避。

新增验证包括：

- 三方实际 proposal/Target 精度、缓存和参考验证配置一致。
- 链和树参考验证器在相同输入下输出及 RNG 状态一致。
- 二词表有限树的全部 posterior 组合穷举，根到出口分布与 AR 条件概率乘积一致。
- 30 组异构 utility/progress/age/预算问题：DP 分配与全部可行分配穷举一致。
- 最终 round 已接受 KV 在输出截断前超过旧 global cap12 容量的回归；旧分配报错，新分配保留全部预期 KV。
- 固定 DDTree cap46 原有容量不变。
- 整段调度器：三档全局预算、C1/C16、每波物理行数不超预算、完整 greedy 输出与 KV/features 长度一致。
- 三种方法在第一个 EOS 截断，不把同一验证波内更晚的 token 泄露到输出。
- T=1 分支追踪：三种方法确实执行共用参考函数；禁止旧 matching/terminal-mass 特殊分支被意外调用。

H20 上的异构请求投毒隔离结果由每个模型的 `request_isolation.json` 保存。
本地有限树 law 测试与请求隔离测试不能代替真实 BF16 执行的完整序列分布认证。

## H20 同版补查

相同四个测试文件在 H20 上核对 SHA256 后运行，环境包含：

```sh
PATH=/root/autodl-tmp/envs/speculative/bin:$PATH
PYTHONPATH=src
```

最终结果：**159 passed，40.67 秒，无跳过**，包括本地需要 CUDA 的13项。
完整日志是同目录 `GPU_TESTS.log`。这仍是上述四个文件的测试范围，不是全仓库验证。

过程记录也保留：

- `GPU_TESTS.first-server-version.log`：158 passed。服务器旧 terminal-mass 测试文件少一个案例；备份后同步为本地同版。
- `GPU_TESTS.path-missing-ninja.log`：158 passed / 1 failed。手动补查命令未带环境 bin PATH，在 C++ 扩展加载阶段找不到已安装的 Ninja，没有进入采样断言。
- 恢复 runner 相同 PATH 后，全套159项通过。未安装依赖、未修改采样源码、未重计或替换性能数据。

两模型各三处真实分歧的后置探针中，所有已选择 query 行的真实 GPU KV 压缩写回
逐字节一致；六处都可在相同 teacher-prefix KV 下观察到 mask/query/forward 导致的
root argmax 翻转。它只隔离这些上下文的有限精度影响，不认证所有实际解码历史。
