from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, write_json
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Diffusion GBV experiments with a fixed evaluation protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "plan", "run", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=ROOT / "configs/gbv_paper_ddtree_counts.json")
        if name != "plan":
            p.add_argument("--data-dir", type=Path, default=ROOT / "datasets/gbv_paper_ddtree_counts")
        if name in {"plan", "run"}:
            p.add_argument("--groups", nargs="+")
            p.add_argument("--only-variants", nargs="+")
            p.add_argument("--output", type=Path, required=name == "run")
        if name == "run":
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--smoke", action="store_true")
            p.add_argument("--profile", action="store_true")
        if name == "audit":
            p.add_argument("--training-jsonl", type=Path, nargs="*", default=[])
            p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "datasets/gbv_paper_ddtree_counts")
    p.add_argument("--code-backend", choices=["docker", "process"])
    p.add_argument("--workers", type=int)
    p.add_argument("--timeout", type=float)
    p.add_argument("--lcb-timeout", type=int)
    p = sub.add_parser("validate-gold")
    p.add_argument("--config", type=Path, default=ROOT / "configs/gbv_paper_ddtree_counts.json")
    p.add_argument("--data-dir", type=Path, default=ROOT / "datasets/gbv_paper_ddtree_counts")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--code-backend", choices=["docker", "process"], default="docker")
    p.add_argument("--timeout", type=float, default=10)
    p = sub.add_parser("report")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--performance-only", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p = sub.add_parser("check-model")
    p.add_argument("--config", type=Path, default=ROOT / "configs/gbv_paper_ddtree_counts.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--code-backend", choices=["docker", "process"], default="docker")
    p.add_argument("--only-variants", nargs="+")
    p = sub.add_parser("export-mtbench")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "datasets/gbv_paper_ddtree_counts")
    p.add_argument("--output", type=Path, required=True)
    for name in ("plan-suite", "run-suite"):
        p = sub.add_parser(name)
        p.add_argument("--suite", type=Path, default=ROOT / "configs/gbv_paper_suite.json")
        p.add_argument("--phase", choices=["gbv-first", "main", "complete"], default="gbv-first")
        p.add_argument("--model-ids", nargs="+")
        p.add_argument("--output", type=Path, required=name == "run-suite")
        if name == "run-suite":
            p.add_argument("--data-dir", type=Path, default=ROOT / "datasets/gbv_paper_ddtree_counts")
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--code-backend", choices=["docker", "process"], default="docker")
    p = sub.add_parser("import-mtbench-judgments")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--judgments", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config) if hasattr(args, "config") else None
    if args.command == "prepare":
        from .data import prepare
        prepare(cfg["datasets"], args.data_dir, cfg.get("evaluation"))
    elif args.command == "plan":
        from .runner import make_plan
        plan = make_plan(cfg, args.groups, args.only_variants)
        if args.output:
            write_json(args.output, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    elif args.command == "run":
        from .runner import run
        run(cfg, args.data_dir, args.output, args.device, args.groups, args.smoke, args.profile, args.only_variants)
    elif args.command == "score":
        from .scoring import score_run
        manifest_path = args.run_dir / "run_manifest.json"
        settings = json.loads(manifest_path.read_text()).get("scoring", {}) if manifest_path.exists() else {}
        score_run(args.run_dir, args.data_dir, args.code_backend or settings.get("code_backend", "docker"),
                  args.workers if args.workers is not None else settings.get("workers", 4),
                  args.timeout if args.timeout is not None else settings.get("timeout_seconds", 10),
                  args.lcb_timeout if args.lcb_timeout is not None else settings.get("lcb_timeout_seconds", 6))
    elif args.command == "report":
        from .report import report
        count = args.bootstrap if args.bootstrap is not None else json.loads((args.run_dir / "run_manifest.json").read_text()).get("bootstrap_samples", 1000)
        print(report(args.run_dir, args.output, count, args.allow_partial, args.performance_only, not args.no_plots))
    elif args.command == "audit":
        from .audit import audit
        print(json.dumps(audit(cfg, args.data_dir, args.training_jsonl, args.output), ensure_ascii=False, indent=2))
    elif args.command == "check-model":
        from .preflight import check_model
        check_model(cfg, args.output, args.device, args.code_backend, args.only_variants)
    elif args.command == "validate-gold":
        from .scoring import validate_gold
        validate_gold(args.data_dir, cfg["datasets"], args.output, args.code_backend, args.timeout, cfg.get("evaluation", {}))
    elif args.command == "export-mtbench":
        from .mtbench import export_answers
        export_answers(args.run_dir, args.data_dir, args.output)
    elif args.command == "import-mtbench-judgments":
        from .mtbench import import_judgments
        import_judgments(args.run_dir, args.export_dir, args.judgments, args.output)
    elif args.command == "plan-suite":
        from .suite import plan_suite
        plan = plan_suite(args.suite, args.phase, args.model_ids)
        if args.output:
            write_json(args.output, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    elif args.command == "run-suite":
        from .suite import run_suite
        run_suite(args.suite, args.data_dir, args.output, args.device, args.code_backend, args.phase, args.model_ids)


if __name__ == "__main__":
    main()
