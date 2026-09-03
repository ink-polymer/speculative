from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
import torch

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def tiny_engine():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from gbv_experiments.engine import Engine, draft_model_class
    torch.manual_seed(876)
    fields = dict(vocab_size=17, hidden_size=32, intermediate_size=48,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                  max_position_embeddings=512, attention_dropout=0, eos_token_id=16)
    target_config = Qwen3Config(num_hidden_layers=3, **fields)
    target_config._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(target_config).eval()
    draft_config = Qwen3Config(num_hidden_layers=1, num_target_layers=3,
                              block_size=4, dflash_config={"target_layer_ids": [1], "mask_token_id": 16}, **fields)
    draft_config._attn_implementation = "sdpa"
    draft = draft_model_class()(draft_config).eval()
    return Engine(target, draft)
