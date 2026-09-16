# 最终复测源码与修正记录

以下文件的本地与 H20 服务器 SHA256 已逐项核对相同。模型配置包含冻结的
Target/Draft revision，`load_models()` 实际传入两者的 revision，并关闭 TF32。

| 文件 | SHA256 |
|---|---|
| scripts/rerun_audited_global_tree_matrix_h20.py | 85b0c697b36d9eae9babaa92eaa888d2f2978bc671651f3fda66bbbb9509bc3f |
| src/gbv_experiments/continuous_tree_block_decode.py | 3f132a41b659d1306cbfc3fbb348c0f89284d097d69851f0f73d8a6e4518587f |
| src/gbv_experiments/engine.py | ad0e99a5d37bdb4dbe104e50e69cddea668c015beb65d1f4d73960d97b74820a |
| src/gbv_experiments/config.py | 433399b7d40a8103682780482a2b6d4885341beab2f4a328ea4a841e61e3e86f |
| scripts/tune_cross_request_architectures.py | eb5951a92f200b6978bd10c0242d1a45c8af3031aa58734e06a6ebb1d67de14f |
| configs/adaptive_block_qwen3_4b.json | b8e0043d0866289aefd8155dc5e5e496689315db7add78ff4a34fa0c85616641 |
| configs/adaptive_block_qwen3_8b.json | 13a4345952fab0acd0d8e14263ad1281f2944c44f27fa626965749011b1dfd04 |

每个模型的 `source_hashes.json` 另存了 sampling、tree 与固定原生 DDTree/DFlash
的 SHA256；运行 manifest 嵌入同份 hash。固定原生代码也已通过其
`SOURCE_SHA256.json` 全部文件清单核对。

## 本轮生产解码源码改动

- `_persistent_cache_capacity()`：容量上界覆盖可扩张全局树以及最终 round 在
  EOS/输出截断前保留的查询 KV。固定 DDTree 原分配不变。
- 新增可显式选择的 `ancestral_reference` 分支：树与链都直接使用同一个
  `tree_verify_ancestral_batched()`，不使用链专用 GPU/CPU round-trip 适配器。
- 原有各生产验证器和默认参数没有替换；最终复测脚本显式选择共同参考分支。
- 没有训练/替换模型权重，没有修改模型配置，没有改旧矩阵原始结果。

## 试跑与最终运行分离

先行的适配器试跑在 H20 上另存为
`/root/autodl-tmp/outputs/audited-global-tree-matrix-adapter-diagnostic-20260916`，
不参与本目录最终表格。最终持久化 screen 使用
`audited-global-tree-matrix-20260916`，原始输出在同名最终 outputs 目录。

修改前的服务器解码源码可恢复副本位于
`/tmp/continuous_tree_block_decode.before-audited-matrix-20260916.py`。
本地已有用户改动未回滚或覆盖为 Git HEAD。
