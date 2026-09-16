# Third-party notices

The original project MIT license and each vendored directory's license are preserved. Upstream models/datasets are not relicensed.

- DDTree pinned and official directories: preserve their LICENSE. Audited baseline executable SHA256 values are in SOURCE_PROVENANCE.json and original manifests.
- DFlash official directory: preserve its LICENSE. Model weights are not included; their own repository terms apply.
- LiveCodeBench official directory: optional inherited evaluation dependency, not exercised by this GSM8K matrix; preserve its LICENSE.
- `data/prepared/gsm8k_{test,train}.jsonl`: prepared public prompts/metadata copied unchanged from the server snapshot. Source: [OpenAI grade-school-math](https://github.com/openai/grade-school-math); preserve [MIT license](data/prepared/GSM8K_LICENSE). Performance uses test offsets0:32; correctness uses96:128. The training file is retained as data provenance, not a claim that DP was trained on it.

Dependencies retain their own licenses. No model checkpoints or authentication material are released.
