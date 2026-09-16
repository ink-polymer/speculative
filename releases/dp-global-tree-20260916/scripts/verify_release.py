"""Verify copied source/artifact hashes and independently recompute cell statistics."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_files():
    return [p for p in ROOT.rglob('*') if p.is_file()
            and not any(part in {'__pycache__', '.pytest_cache', '.git'} or part.endswith('.egg-info') for part in p.relative_to(ROOT).parts)
            and p.suffix != '.pyc']


def verify(record_packaging=False, seal=False):
    provenance_path = ROOT / 'SOURCE_PROVENANCE.json'
    provenance = json.loads(provenance_path.read_text())
    if record_packaging:
        entry = provenance['files']['pyproject.toml']
        entry['release_sha256'] = sha(ROOT / 'pyproject.toml')
        entry['release_edit'] = 'Packaging-only rename/description and removal of excluded CLI entrypoints; original hash retained; no decoder change'
        provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    for relative, entry in provenance['files'].items():
        assert sha(ROOT / relative) == entry.get('release_sha256', entry['sha256']), relative
    assert sha(ROOT / 'src/gbv_experiments/continuous_tree_block_decode.py') == \
        '3f132a41b659d1306cbfc3fbb348c0f89284d097d69851f0f73d8a6e4518587f'
    audit = ROOT / 'results/pilots/20260916/audited_global_tree_matrix_h20'
    summary = json.loads((ROOT / 'EVIDENCE_SUMMARY.json').read_text())
    cells = {(c['model'], c['temperature'], c['budget'], c['concurrency']): c for c in summary['cells']}
    names = ('ddtree_full46', 'dflash_r16', 'ours_global_hetero_11_23_45_h20_curve384_tierfit_a8_c1_propfp32')
    paired = 0
    for model in ('qwen3_4b', 'qwen3_8b'):
        for relative, expected in json.loads((audit / model / 'source_hashes.json').read_text()).items():
            assert sha(ROOT / relative) == expected, (model, relative)
        for temperature in (0, 1):
            for budget in (96, 193, 384):
                run = audit / model / f't{temperature}_r{budget}'
                records = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines()]
                assert len(records) == 16
                paired += len(records)
                analysis = json.loads((run / 'analysis.json').read_text())
                assert set(analysis['by_concurrency']) == {'1', '4', '8', '16', '32'}
                for concurrency, item in analysis['by_concurrency'].items():
                    cell = cells[(model, temperature, budget, int(concurrency))]
                    methods = item['methods']
                    selected = [r for r in records if r['concurrency'] == int(concurrency)]
                    for name in names:
                        for record in selected:
                            samples = record['samples'][name]
                            assert len(samples) == 3
                            for field in ('wall_ms', 'output_tokens'):
                                assert record['methods'][name][field] == statistics.median(s[field] for s in samples)
                            assert all(o == record['outputs'][name][0] for o in record['outputs'][name])
                        raw_tokens = sum(r['methods'][name]['output_tokens'] for r in selected)
                        raw_wall = sum(r['methods'][name]['wall_ms'] for r in selected)
                        assert math.isclose(1000 * raw_tokens / raw_wall,
                                            methods[name]['tokens_per_second'], rel_tol=1e-14)
                    assert cell['dp_tok_s'] == methods[names[2]]['tokens_per_second']
                    for field, method in [('dp_over_ddtree', names[0]), ('dp_over_dflash', names[1])]:
                        expected = methods[names[2]]['tokens_per_second'] / methods[method]['tokens_per_second']
                        assert math.isclose(cell[field], expected, rel_tol=1e-14)
    assert len(cells) == 60 and paired == 192
    for label, group in summary['groups'].items():
        selected = [c for c in cells.values()
                    if ('t1' not in label or c['temperature'] == 1)
                    and ('concurrency_ge4' not in label or c['concurrency'] >= 4)]
        assert len(selected) == group['cells']
        for metric in ('dp_over_ddtree', 'dp_over_dflash'):
            values = [c[metric] for c in selected]
            actual = math.exp(sum(math.log(v) for v in values) / len(values))
            assert math.isclose(actual, group[metric]['geometric_mean'], rel_tol=1e-14)
            assert group[metric]['wins_strict_gt1'] == sum(v > 1 for v in values)
    # Credential scan reports paths only and never prints matched values.
    patterns = {
        'private_key': rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',
        'github_token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b',
        'credential_assignment': rb'''(?i)(?:password|passwd|api_key|access_token)\s*[:=]\s*["'][A-Za-z0-9_./+-]{8,}["']''',
        'mixed_12char_identifier': rb'(?<![A-Za-z0-9])(?=[A-Za-z0-9]{12}(?![A-Za-z0-9]))(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{12}',
    }
    for path in payload_files():
        for category, pattern in patterns.items():
            matches = set(re.findall(pattern, path.read_bytes()))
            if category == 'mixed_12char_identifier':
                matches -= {b'Qwen3RMSNorm'}  # verified upstream public class-name false positive
            assert not matches, ('potential credential; inspect locally without publishing values', str(path), category)
    manifest_path = ROOT / 'RELEASE_MANIFEST.json'
    if seal:
        entries = {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                   for p in sorted(payload_files()) if p != manifest_path}
        manifest_path.write_text(json.dumps({'files': entries, 'scope': 'publication payload excluding this manifest and runtime caches'}, indent=2) + '\n')
    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text())['files']
        actual_paths = {str(p.relative_to(ROOT)) for p in payload_files() if p != manifest_path}
        assert set(entries) == actual_paths
        for relative, entry in entries.items():
            assert sha(ROOT / relative) == entry['sha256'] and (ROOT / relative).stat().st_size == entry['bytes'], relative
    print(json.dumps({'status': 'passed', 'copied_original_files_verified': len(provenance['files']),
                      'core_and_six_recorded_source_hashes': 'match', 'configurations': len(cells),
                      'paired_records': paired, 'summary_recomputed_from_raw_records': True,
                      'credential_patterns': 'no unresolved matches', 'sealed_manifest': manifest_path.exists()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-packaging-edit', action='store_true')
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    verify(args.record_packaging_edit, args.seal)
