from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from dflash_specblock.ddtree_builder import DDTreeBuilder
from dflash_specblock.paper.common import ROOT, VARIANTS, atomic_json, load_json
from dflash_specblock.paper.controller import PaperAdaptiveBuilder
from dflash_specblock.paper.official import main
from dflash_specblock.paper.official_spec import BUDGETS, COMMIT, LIMITS, MODELS, SOURCES, UPSTREAM, data_utils, load_config, upstream, verify_sources
from dflash_specblock.paper.official_data import check_manifest, prepare, select_official
from dflash_specblock.paper.official_reporting import official_rows, validate_pair
from dflash_specblock.paper.official_worker import audit_response, method_names

torch.set_num_threads(1)


def cfg():
    return load_config(ROOT/"configs/paper_t0_full.json")


def test_official_matrix_is_extracted_from_pinned_script_and_cli(capsys):
    shell = (UPSTREAM/"run_benchmark.sh").read_text()
    assert dict((k,int(v)) for k,v in re.findall(r'"([a-z0-9-]+):(\d+)"',shell)) == LIMITS
    pairs_block = shell.split("MODEL_DRAFT_PAIRS=(",1)[1].split(")",1)[0]
    assert [tuple(x) for x in re.findall(r'"([^"|]+)\|([^"]+)"',pairs_block)] == MODELS
    assert "--max-new-tokens 2048" in shell
    code = (UPSTREAM/"benchmark.py").read_text()
    assert 'default="16,32,64,128,256,512,1024"' in code
    assert cfg()["tree_budgets"] == BUDGETS
    main(["plan"])
    result = json.loads(capsys.readouterr().out)
    assert result["cases"] == 1072 and result["turns_per_method"] == 1152
    assert result["generation_calls"] == 55296 and result["nproc_per_node"] == 8
    assert result["official_samples"] and not result["full_split"] and not result["training"]
    assert len(result["models"]) == 3


def test_source_integrity_and_original_builder_byte_identity():
    assert verify_sources()["commit"] == COMMIT
    core = ROOT/"src/dflash_specblock/ddtree_builder.py"
    assert hashlib.sha256(core.read_bytes()).hexdigest() == "b3d8b79b993f4507ba593741386ee875999fae3a5a3de07a5878528ff9167246"


def test_selection_is_identical_to_hf_official_rule():
    from datasets import Dataset
    data = Dataset.from_list([{"x":i} for i in range(200)])
    assert select_official(data,128)["x"] == data.shuffle(seed=0).select(range(128))["x"]
    assert select_official(data,200)["x"] == list(range(200))


def test_prepare_executes_official_prompts_and_freezes_all_sources(tmp_path, monkeypatch):
    from datasets import Dataset, disable_progress_bars
    disable_progress_bars()
    u = data_utils()
    calls = []
    def loader(path,*args,**kwargs):
        calls.append((path,args,copy.deepcopy(kwargs)))
        n = 30 if path in {"HuggingFaceH4/aime_2024","MathArena/aime_2025"} else 80 if path=="HuggingFaceH4/mt_bench_prompts" else 200
        result = Dataset.from_list([{"question":f"q{i}", "problem":f"p{i}",
            "prompt":[f"first{i}",f"second{i}"] if path=="HuggingFaceH4/mt_bench_prompts" else f"code{i}",
            "question_content":f"lcb{i}", "starter_code":"def solve(): pass" if i%2 else "",
            "problem_statement":f"issue{i}", "instruction":f"instruction{i}",
            "input":f"input{i}" if i%2 else ""} for i in range(n)])
        return {"test":result} if path=="json" else result
    monkeypatch.setattr(u,"load_dataset",loader)
    monkeypatch.setattr("huggingface_hub.HfApi",lambda:SimpleNamespace(
        dataset_info=lambda _:SimpleNamespace(sha="a"*40),
        model_info=lambda _:SimpleNamespace(sha="b"*40)))
    manifest = prepare(tmp_path)
    assert sum(v["rows"] for v in manifest["files"].values()) == 1072
    assert not manifest["training"] and not manifest["full_split"]
    assert prepare(tmp_path) == manifest
    assert ("google-research-datasets/mbpp",("sanitized",),{"split":"test","revision":"a"*40}) in calls
    assert any(p=="HuggingFaceH4/mt_bench_prompts" and k["split"]=="train" for p,a,k in calls)
    lcb = next(k for p,a,k in calls if p=="json")
    assert len(lcb["data_files"]["test"])==6 and all("/resolve/"+("a"*40)+"/" in url for url in lcb["data_files"]["test"])
    rows = load_json(tmp_path/"livecodebench.json")
    assert rows[0]["turns"][0].startswith("You are an expert Python programmer. You will be given a question")
    assert rows[0]["turns"][0].endswith("### Answer: (use the provided format with backticks)")
    assert load_json(tmp_path/"mt-bench.json")[0]["turns"] == ["first0","second0"]
    assert all(len(r["turns"])==2 for r in load_json(tmp_path/"mt-bench.json"))
    changed = load_json(tmp_path/"gsm8k.json")
    changed[0]["turns"] = ["tampered"]
    atomic_json(tmp_path/"gsm8k.json",changed)
    with pytest.raises(ValueError,match="Dataset file changed"):
        check_manifest(tmp_path)


@pytest.mark.parametrize("seed",range(4))
def test_official_tree_conversion_matches_unmodified_builder(seed,monkeypatch):
    from dflash_specblock.paper.adaptive_official import build_with_controller
    u = upstream()
    monkeypatch.setattr(u.ddtree,"cuda_time",time.perf_counter)
    logits = torch.randn(15,160,generator=torch.Generator().manual_seed(seed))
    expected = u.ddtree.build_ddtree_tree(logits,60)
    actual = build_with_controller(logits,DDTreeBuilder(15,60))
    for i in (0,1,4):
        assert torch.equal(expected[i],actual[i])
    assert expected[2:4] == actual[2:4]


@pytest.mark.parametrize("limit,stops",[(1,[]),(17,[]),(49,[]),(49,[9]),(49,[6]),(17,[22])])
def test_official_adaptive_loop_matches_official_ar_on_mock(limit,stops,monkeypatch):
    from test_integration_speedup import _MockDraft, _MockTarget, _make_embedding, _make_lm_head
    from dflash_specblock.paper.adaptive_official import adaptive_generate
    u = upstream()
    monkeypatch.setattr(u.dflash,"cuda_time",time.perf_counter)
    monkeypatch.setattr(u.ddtree,"cuda_time",time.perf_counter)
    monkeypatch.setattr(u.ddtree,"_CPP_COMPACT_ENABLED",False)
    embed, head = _make_embedding(),_make_lm_head()
    target = _MockTarget(embed,head)
    target.model = SimpleNamespace(embed_tokens=embed)
    target.dtype, target.device = torch.float32,torch.device("cpu")
    draft = _MockDraft(embed,head)
    draft.device = torch.device("cpu")
    kwargs = dict(model=draft,target=target,input_ids=torch.tensor([[5]]),
        mask_token_id=draft.mask_token_id,max_new_tokens=limit,stop_token_ids=stops,temperature=0.)
    ar = u.dflash.dflash_generate(**kwargs,block_size=1)
    for variant in VARIANTS:
        builder = PaperAdaptiveBuilder(cfg()["adaptive"],variant)
        result = adaptive_generate(**kwargs,block_size=16,builder=builder)
        assert torch.equal(result.output_ids,ar.output_ids)
        assert result.time_per_output_token > 0 and result.adaptive_decisions
        assert all(d["tree_nodes"] in cfg()["adaptive"]["budget_candidates"] for d in result.adaptive_decisions)


def response(tpot,token=7):
    return SimpleNamespace(time_per_output_token=tpot,acceptance_lengths=[2,4],
                           output_ids=torch.tensor([[1,token]]),num_input_tokens=1,num_output_tokens=1)


def test_official_mean_tpot_not_ratio_of_total_times():
    sdpa = {"target_attn_implementation":"sdpa","responses":[]}
    fa = {"target_attn_implementation":"flash_attention_2","responses":[]}
    for base in (10.,2.):
        sdpa["responses"].append({"baseline":response(base),"dflash":response(2.),
            **{f"ddtree_tb{b}":response(3. if b!=128 else 1.5) for b in BUDGETS},
            **{v:response(1.) for v in VARIANTS}})
        fa["responses"].append({"baseline":response(base/2),"dflash":response(1.8)})
    rows = {r["method"]:r for r in official_rows(sdpa,fa,VARIANTS)}
    assert rows["adaptive"]["speedup_vs_target"] == 3.
    assert rows["adaptive"]["speedup_vs_best_ddtree"] == 1.5
    assert rows["DDTree-best"]["selected_key"] == "ddtree_tb128"
    assert rows["DFlash"]["method_backend"] == "flash_attention_2"
    assert rows["adaptive"]["target_baseline_backend"] == "flash_attention_2"


def test_mismatch_audit_is_fail_closed(tmp_path):
    with pytest.raises(RuntimeError,match="Greedy mismatch"):
        audit_response({"baseline":response(1,7),"adaptive":response(.5,8)}, index=0,turn=0,
                       input_ids=torch.tensor([[1]]),diagnostic_path=tmp_path/"failed.json")
    assert load_json(tmp_path/"failed.json")["mismatching_tokens"] == {"adaptive":[8]}


def test_official_method_order_and_no_t1_support():
    assert method_names("sdpa",VARIANTS)[:9] == ["baseline","dflash"]+[f"ddtree_tb{b}" for b in BUDGETS]
    assert method_names("flash_attention_2",VARIANTS) == ["baseline","dflash"]


def synthetic_environment():
    return {"cuda":"synthetic", "gpu":"synthetic", "nproc_per_node":1, "flash_attn":"test-only",
            "benchmark_gpus":[{"rank":0,"gpu":"synthetic","uuid":"synthetic"}]}


def synthetic_run(backend="sdpa"):
    methods = method_names(backend, VARIANTS)
    return {"target_attn_implementation":backend, "draft_attn_implementation":"flash_attention_2",
        "args":{"dataset":"gsm8k", "model_name_or_path":MODELS[0][0], "draft_name_or_path":MODELS[0][1],
                "temperature":0., "max_samples":128, "max_new_tokens":2048,
                "tree_budget":",".join(map(str,BUDGETS)), "flash_attn":backend!="sdpa"},
        "methods":methods, "block_size":16, "smoke":False, "source_lock":{"test_only":True},
        "hardware":[{"rank":0,"gpu":"synthetic","uuid":"synthetic","flash_attn":"test-only"}], "world_size":1,
        "responses":[{**{m:response(2. if m=="baseline" else 1.) for m in methods},
                      "_audit":{"index":0,"turn":0,"exact_match":True,"input_sha256":"synthetic"}}]}


@pytest.mark.parametrize("field,value",[("source_lock",{}),("world_size",8),("block_size",32),
    ("smoke",True),("max_new_tokens",128),("max_samples",1319),("tree_budget","60"),
    ("flash_attn",True),("hardware",[{"rank":1}])])
def test_run_contract_rejects_mixed_protocol_fields(field,value):
    from dflash_specblock.paper.official_reporting import validate_run_contract
    run = synthetic_run()
    validate_run_contract(run,{"test_only":True},1,0,synthetic_environment())
    if field in run["args"]:
        run["args"][field] = value
    else:
        run[field] = value
    with pytest.raises(ValueError,match="differ from contract"):
        validate_run_contract(run,{"test_only":True},1,0,synthetic_environment())


def test_cross_backend_tokens_inputs_and_missing_turn_are_fail_closed():
    expected = [{"index":0,"turns":["synthetic prompt"]}]
    sdpa, fa = synthetic_run(),synthetic_run("flash_attention_2")
    validate_pair(sdpa,fa,"gsm8k",0,VARIANTS,expected)
    fa["responses"][0]["_audit"]["input_sha256"] = "changed"
    with pytest.raises(ValueError,match="inputs or greedy outputs"):
        validate_pair(sdpa,fa,"gsm8k",0,VARIANTS,expected)
    fa = synthetic_run("flash_attention_2")
    fa["responses"][0]["dflash"] = response(1.,8)
    with pytest.raises(ValueError,match="Invalid official tokens"):
        validate_pair(sdpa,fa,"gsm8k",0,VARIANTS,expected)
    with pytest.raises(ValueError,match="Incomplete official sampled"):
        validate_pair(sdpa,synthetic_run("flash_attention_2"),"gsm8k",0,VARIANTS,
                      [{"index":0,"turns":["first","second"]}])


def test_summary_checks_contract_completion_environment_and_artifact_hash(tmp_path,monkeypatch):
    from dflash_specblock.paper import official_reporting as reporting
    from dflash_specblock.paper.common import contract, file_hash
    run_dir, data_dir = tmp_path/"run",tmp_path/"data"
    manifest = {"test_only":True}
    config = cfg()
    metadata = {"config":config,"model_indices":[0],"datasets":["gsm8k"],"smoke_count":0,
        "dataset_manifest":manifest,"source_manifest":verify_sources(),"nproc_per_node":1}
    identity = contract(run_dir,metadata)
    atomic_json(run_dir/"environment.json",synthetic_environment())
    atomic_json(data_dir/"source_revisions.json",{"test_only":True})
    atomic_json(data_dir/"gsm8k.json",[{"index":0,"turns":["synthetic prompt"]}])
    monkeypatch.setattr(reporting,"check_manifest",lambda _:manifest)
    for backend in ("sdpa","flash_attention_2"):
        path = run_dir/(reporting.run_stem("gsm8k",0,backend)+".pt")
        run = {**synthetic_run(backend),"protocol_identity":identity}
        torch.save(run,path)
        atomic_json(path.with_suffix(".complete.json"),{"identity":identity,"sha256":file_hash(path),
            "turns":1,"cases":1,"methods":run["methods"],"smoke":False})
    reporting.summarize(run_dir,data_dir,config,identity,[0],["gsm8k"])
    report = load_json(run_dir/"tables.json")
    assert report["protocol_identity"] == identity and report["environment_sha256"]
    assert not report["full_official_t0_model_dataset_matrix"]
    assert (run_dir/"tables.csv").exists() and (run_dir/"tables.md").exists()
    marker = path.with_suffix(".complete.json")
    completion = load_json(marker)
    atomic_json(marker,{**completion,"cases":2})
    with pytest.raises(ValueError,match="identity/count mismatch"):
        reporting.load_completed(path,identity)
    atomic_json(marker,{**completion,"sha256":"changed"})
    with pytest.raises(ValueError,match="hash/identity changed"):
        reporting.load_completed(path,identity)
    atomic_json(run_dir/"environment.json",{"cuda":None,"gpu":None,"nproc_per_node":1})
    with pytest.raises(ValueError,match="GPU environment"):
        reporting.summarize(run_dir,data_dir,config,identity,[0],["gsm8k"])


@pytest.mark.parametrize("model_index", [0, 1])
def test_worker_multiturn_keeps_official_history_method_and_run_completion(tmp_path,monkeypatch,model_index):
    import transformers
    from dflash_specblock.paper import official_worker as worker_module, adaptive_official
    seen = []
    loaded = []
    class FakeModel:
        block_size, mask_token_id = 16,9999
        device = torch.device("cpu")
        def to(self,*args): return self
        def eval(self): return self
        @classmethod
        def from_pretrained(cls,*args,**kwargs):
            loaded.append((args[0],kwargs["revision"]))
            return cls()
    class Tokenizer:
        eos_token_id = 9998
        def apply_chat_template(self,messages,**kwargs):
            seen.append(copy.deepcopy(messages))
            return json.dumps(messages)
        def encode(self,text,**kwargs): return torch.tensor([[1]])
        def decode(self,tokens,**kwargs): return f"token-{tokens[0]}"
    def generated(token):
        return response(1.,token)
    dist = SimpleNamespace(init=lambda:None,local_rank=lambda:0,rank=lambda:0,size=lambda:1,
                           all_gather=lambda x:[x],is_main=lambda:True)
    fake = SimpleNamespace(dist=dist,model=SimpleNamespace(DFlashDraftModel=FakeModel),
        dflash=SimpleNamespace(dflash_generate=lambda **kw:generated(1 if kw["block_size"]==1 else 2)),
        ddtree=SimpleNamespace(maybe_enable_cpp_compact=lambda _:None,load_cpp_compact_module=lambda:object(),
                              ddtree_generate=lambda **kw:generated(kw["tree_budget"])))
    monkeypatch.setattr(worker_module,"upstream",lambda:fake)
    monkeypatch.setattr(worker_module,"check_manifest",lambda _: {})
    monkeypatch.setattr(transformers,"AutoModelForCausalLM",FakeModel)
    monkeypatch.setattr(transformers.AutoTokenizer,"from_pretrained",lambda *a,**k:Tokenizer())
    monkeypatch.setattr(adaptive_official,"adaptive_generate",lambda **kw:generated(999))
    monkeypatch.setitem(sys.modules,"flash_attn",SimpleNamespace(__version__="test-only"))
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:None)
    monkeypatch.setattr(torch.cuda,"manual_seed_all",lambda _:None)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda _: "synthetic")
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda _:SimpleNamespace(uuid="synthetic"))
    # Deliberately differing synthetic tokens isolate the history routing test.
    # Real audit_response fail-closed behavior is tested separately above.
    monkeypatch.setattr(worker_module,"audit_response",lambda response,**kw:
                        {"index":kw["index"],"turn":kw["turn"],"input_sha256":"test","exact_match":True})
    atomic_json(tmp_path/"source_revisions.json",{"models":{v:"a"*40 for v in MODELS[model_index]}})
    atomic_json(tmp_path/"mt-bench.json",[{"index":0,"turns":["first","second"]}])
    atomic_json(tmp_path/"environment.json",synthetic_environment())
    args = SimpleNamespace(data_dir=tmp_path,model_index=model_index,backend="sdpa",dataset="mt-bench",
                           smoke_count=1,output=tmp_path/"case.pt",identity="synthetic",run_dir=tmp_path)
    worker_module.worker(args,cfg())
    assert loaded == [(name,"a"*40) for name in MODELS[model_index]]
    assert seen[0] == [{"role":"user","content":"Warmup"}]
    assert seen[-1][1] == {"role":"assistant","content":"token-1024"}
    assert all("token-999" not in str(m) for m in seen)
    saved = torch.load(args.output,weights_only=False)
    assert len(saved["responses"]) == 2 and saved["smoke"]
    assert args.output.with_suffix(".complete.json").exists()
