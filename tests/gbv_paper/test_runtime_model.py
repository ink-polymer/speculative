from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from gbv_experiments.runtime_model import (MODEL_PARAMETERS_SCHEMA,
                                           RUNTIME_READY_GATE_SCHEMA,
                                           enforce_runtime_model_gate,
                                           is_registered_same_tree_t1,
                                           runtime_model_expectations,
                                           strict_runtime_model_expectations,
                                           validate_model_parameters_evidence,
                                           validate_runtime_model_identity)


def fake_model(dtype=torch.bfloat16, attention="sdpa"):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)).to(dtype)
    model.eval().requires_grad_(False)
    model.config = SimpleNamespace(
        _attn_implementation=attention,
        _name_or_path="fixture/model",
        _commit_hash="a" * 40,
        model_type="fixture",
        vocab_size=17,
        hidden_size=4,
        num_hidden_layers=2,
    )
    return model


def fake_engine(dtype=torch.bfloat16, attention="sdpa"):
    draft = fake_model(dtype, attention)
    draft.block_size = 16
    draft.target_layer_ids = [0, 1]
    draft.mask_token_id = 16
    return SimpleNamespace(
        target=fake_model(dtype, attention),
        draft=draft,
        proposal_adapter=None,
    )


@pytest.fixture(autouse=True)
def disable_tf32(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


def test_strict_runtime_identity_is_serializable_and_records_actual_state():
    identity = enforce_runtime_model_gate(fake_engine())
    assert json.loads(json.dumps(identity)) == identity
    assert identity["target"]["floating_parameter_dtypes"] == ["torch.bfloat16"]
    assert identity["draft"]["attention_implementation"] == "sdpa"
    assert identity["draft_runtime"] == {
        "block_size":16, "target_layer_ids":[0, 1], "mask_token_id":16,
    }
    assert identity["tf32"] == {"cuda_matmul": False, "cudnn": False}
    assert identity["proposal_adapter_attached"] is False


def test_terminal_formal_reuses_the_strict_runtime_gate():
    from gbv_experiments.terminal_formal import model_gate

    assert model_gate(fake_engine())["target"]["attention_implementation"] == "sdpa"


@pytest.mark.parametrize("drift", [
    "dtype", "attention", "training", "requires_grad", "cuda_tf32", "cudnn_tf32",
    "adapter",
])
def test_strict_runtime_gate_fails_closed_on_each_state_drift(drift, monkeypatch):
    engine = fake_engine()
    if drift == "dtype":
        engine.target.to(torch.float32)
    elif drift == "attention":
        engine.draft.config._attn_implementation = "eager"
    elif drift == "training":
        engine.target.train()
    elif drift == "requires_grad":
        next(engine.draft.parameters()).requires_grad_(True)
    elif drift == "cuda_tf32":
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    elif drift == "cudnn_tf32":
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    else:
        engine.proposal_adapter = object()
    with pytest.raises(RuntimeError, match="Runtime model gate failed"):
        enforce_runtime_model_gate(engine)


def test_runtime_gate_checks_every_submodule_is_in_eval_mode():
    engine = fake_engine()
    engine.target[0].train()
    assert engine.target.training is False
    with pytest.raises(RuntimeError, match="eval state"):
        enforce_runtime_model_gate(engine)


def test_runtime_gate_rejects_missing_float_parameters_and_adapter_contract():
    engine = fake_engine()
    engine.target = torch.nn.Identity().eval()
    with pytest.raises(RuntimeError, match="no floating-point parameters"):
        enforce_runtime_model_gate(engine)
    del engine.proposal_adapter
    engine.target = fake_model()
    with pytest.raises(RuntimeError, match="proposal adapter"):
        enforce_runtime_model_gate(engine)


def test_generic_runner_expectations_follow_registered_model_configuration(monkeypatch):
    expected = runtime_model_expectations({
        "dtype": "float32",
        "target_attention": "eager",
        "draft_attention": "eager",
        "allow_tf32": True,
    })
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    identity = enforce_runtime_model_gate(fake_engine(torch.float32, "eager"), expected)
    assert identity["target"]["floating_parameter_dtypes"] == ["torch.float32"]


def test_registered_same_tree_t1_cannot_relax_the_strict_runtime_contract():
    cfg = {"explicit_variants": [{"variant": {
        "method": "ddtree_fused_scan", "temperature": 1.0,
    }}]}
    assert is_registered_same_tree_t1(cfg)
    assert strict_runtime_model_expectations()["strict_same_tree_t1"] is False
    assert strict_runtime_model_expectations()["floating_parameter_dtype"] == (
        "torch.bfloat16"
    )
    with pytest.raises(RuntimeError, match="same-tree T=1"):
        runtime_model_expectations({"dtype": "float32"}, strict_same_tree_t1=True)


def registered_runtime_evidence():
    model = {
        "target":"Qwen/Qwen3-4B",
        "target_revision":"1cfa9a7208912126459214e8b04321603b3df60c",
        "draft":"z-lab/Qwen3-4B-DFlash-b16",
        "draft_revision":"b74e3a329c4d963783143b1e970d95b002be72bd",
        "dtype":"bfloat16", "target_attention":"sdpa",
        "draft_attention":"sdpa", "allow_tf32":False,
    }
    expected = runtime_model_expectations(
        model, strict_same_tree_t1=True, device="cuda:0",
    )
    identity = enforce_runtime_model_gate(fake_engine())
    for role in ("target", "draft"):
        contract = expected[role + "_config"]
        runtime = identity[role]
        runtime["class"] = contract["class"]
        runtime["parameter_tensors"] = contract["parameter_tensors"]
        runtime["floating_parameter_tensors"] = contract["parameter_tensors"]
        runtime["parameters"] = contract["parameters"]
        runtime["parameter_devices"] = ["cuda:0"]
        runtime["config"] = {
            "_name_or_path":contract["name_or_path"],
            "_commit_hash":contract["commit_hash"],
            "model_type":contract["model_type"],
            "vocab_size":contract["vocab_size"],
            "hidden_size":contract["hidden_size"],
            "num_hidden_layers":contract["num_hidden_layers"],
        }
    identity["draft_runtime"] = deepcopy(expected["draft_runtime"])
    identity["tokenizer"] = deepcopy(expected["tokenizer"]) | {
        "minimum_token_id":0,
        "maximum_token_id":151668,
    }
    validate_runtime_model_identity(identity, expected)
    parameters = {
        "schema":MODEL_PARAMETERS_SCHEMA,
        "run_id":"a" * 64,
        "target_parameters":identity["target"]["parameters"],
        "draft_parameters":identity["draft"]["parameters"],
        "trainable_parameters":0,
        "draft_block_size":identity["draft_runtime"]["block_size"],
        "target_feature_layers":identity["draft_runtime"]["target_layer_ids"],
        "runtime_model_gate":{
            "schema":RUNTIME_READY_GATE_SCHEMA,
            "status":"ready_for_timing",
            "expectations":deepcopy(expected),
            "identity_after_load":identity,
            "identity_before_timing":identity,
            "passed_after_load":True,
            "passed_before_timing":True,
            "identities_match":True,
        },
    }
    validate_model_parameters_evidence(
        parameters, run_id="a" * 64, expected=expected, require_ready=True,
    )
    return expected, identity, parameters


@pytest.mark.parametrize("field,value", [
    ("block_size", 15),
    ("target_layer_ids", [0, 1, 2, 3, 4]),
    ("mask_token_id", 151668),
])
def test_registered_runtime_gate_binds_official_draft_runtime(field, value):
    expected, identity, _ = registered_runtime_evidence()
    identity["draft_runtime"][field] = value
    with pytest.raises(RuntimeError, match="Draft registered runtime"):
        validate_runtime_model_identity(identity, expected)


def test_registered_runtime_gate_rejects_nonfloating_checkpoint_parameters():
    expected, identity, _ = registered_runtime_evidence()
    identity["target"]["floating_parameter_tensors"] -= 1
    with pytest.raises(RuntimeError, match="non-floating"):
        validate_runtime_model_identity(identity, expected)


@pytest.mark.parametrize("location", [
    "parameters_schema", "gate_schema", "identity_schema", "identity_minimum",
    "expectation_count",
])
def test_persisted_runtime_gate_rejects_json_numeric_type_spoofing(location):
    expected, _, parameters = registered_runtime_evidence()
    if location == "parameters_schema":
        parameters["schema"] = True
    elif location == "gate_schema":
        parameters["runtime_model_gate"]["schema"] = 1.0
    elif location == "identity_schema":
        parameters["runtime_model_gate"]["identity_after_load"]["schema"] = 2.0
    elif location == "identity_minimum":
        parameters["runtime_model_gate"]["identity_after_load"]["tokenizer"][
            "minimum_token_id"
        ] = False
    else:
        parameters["runtime_model_gate"]["expectations"]["target_config"][
            "parameters"
        ] = float(expected["target_config"]["parameters"])
    with pytest.raises(RuntimeError):
        validate_model_parameters_evidence(
            parameters, run_id="a" * 64,
            expected=expected, require_ready=True,
        )
