# Third-party sources

The portable experiment bundle includes the DFlash model implementation copied
from this workspace's `third_party/ddtree_official/model/` directory, with its
existing MIT license at `third_party/ddtree_official/LICENSE` (copyright 2026
Liran Ringel). Only the model files needed by this experiment runner are bundled.
The originating repository is https://github.com/liranringel/ddtree and the
workspace's recorded checkout is `c96427a185677bf4133ed865dd1626a5041aef9b`.
The bundle's `SOURCE_SHA256.json` identifies the actual shipped files.

The local DFlash baseline uses greedy draft tokens and target-sampling prefix
matching, following `dflash.py` in the pinned DDTree checkout above. Its shared
benchmark runtime and timing differ from the optimized upstream implementation.
The separate single-path p/q rejection baseline is labeled as a mechanism control.

DFlash model/paper: https://github.com/z-lab/dflash and
https://arxiv.org/abs/2602.06036 .

The new sampling kernels implement the GBV selection/reweighting construction
of Thomas and Pal (https://arxiv.org/abs/2602.16961) and the block-verification
construction cited by that paper. They use a scaled polynomial evaluation of
the CDF difference. Algorithmic attribution remains with the respective authors.

The unmodified LiveCodeBench execution utility is bundled at
`third_party/livecodebench_official/testing_util.py`, with its upstream MIT
license in the same directory. Source repository:
https://github.com/LiveCodeBench/LiveCodeBench ; pinned commit:
`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`; upstream file:
`lcb_runner/evaluation/testing_util.py`. The local adapter provides subprocess
or container isolation and preserves the utility's test semantics. Its resource
limits and Python environment are part of this experiment protocol.

MT-Bench questions are downloaded separately from `lm-sys/FastChat`, commit
`587d5cfa1609a43d192cedb8441cac3c17db105d`, file
`fastchat/llm_judge/data/mt_bench/question.jsonl`:
https://github.com/lm-sys/FastChat/tree/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge .
The offline answer export and judgment import use FastChat's single-answer
evaluation format. They do not bundle judge-model responses or call an API.

Public model checkpoints and benchmark data are downloaded separately, are not
contained in this code bundle, and retain their original licenses and notices.
