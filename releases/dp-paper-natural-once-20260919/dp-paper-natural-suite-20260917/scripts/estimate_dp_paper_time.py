"""Measured pilot projection with explicit unmeasured-condition uncertainty."""
import argparse
import json
import math
from pathlib import Path
import statistics

from run_dp_paper_study import cases_for,methods_for

PHASES=['main','budget','ablation','temperature0','long_output','context','quality_math']


def estimate(root):
    observations={};measured={};by_model={};coverage={}
    for model in ['qwen3_4b','qwen3_8b']:
        paths=sorted((root/'pilot'/model/'groups').glob('*.json'))
        measured[model]=0
        for path in paths:
            record=json.loads(path.read_text());case=record['identity']['case']
            for run in record['runs']:
                key=(model,case['concurrency'],case['max_new_tokens'],run['method'])
                output=sum(map(len,run['outputs']));n=len(run['outputs']);seconds=run['outer_seconds']
                observations.setdefault(key,[]).append({'tokens_per_second':output/seconds,
                    'mean_output_tokens':output/n,'seconds':seconds,'prompt_tokens':statistics.mean(record['actual_prompt_tokens'])})
                measured[model]+=seconds
        coverage[model]={'pilot_groups_observed':len(paths),'pilot_complete':(root/'pilot'/model/'complete.json').exists()}
    if not all(coverage[m]['pilot_complete'] for m in coverage):
        raise RuntimeError(f'Both complete pilots required; current coverage: {coverage}')
    for model in coverage:
        stages={}
        for phase in PHASES:
            projected=cap_projection=0;calls=0;interpolated=0;unmeasured=0
            for case in cases_for(phase):
                c=case['concurrency'];cap=case['max_new_tokens']
                for name in methods_for(case):
                    base=name if name in {'ar','dflash','ddtree','dp'} else 'dp'
                    pilot_cap=128 if cap<=128 else 512
                    def obs(concurrency):
                        rows=observations[(model,concurrency,pilot_cap,base)]
                        return {key:statistics.median(r[key] for r in rows) for key in rows[0]}
                    if c==8:
                        left,right=obs(4),obs(16)
                        rate=math.sqrt(left['tokens_per_second']*right['tokens_per_second'])
                        length=(left['mean_output_tokens']+right['mean_output_tokens'])/2
                        prompt_length=(left['prompt_tokens']+right['prompt_tokens'])/2
                        interpolated+=1
                    else:
                        row=obs(c);rate=row['tokens_per_second'];length=row['mean_output_tokens'];prompt_length=row['prompt_tokens']
                    # EOS-limited estimate versus every request reaching its cap.
                    expected_tokens=case['requests']*min(cap,length)
                    cap_tokens=case['requests']*cap
                    factor=1.0
                    if case['prompt_tokens'] is not None:
                        # Unmeasured synthetic-context scaling, never reported as a measurement.
                        factor=max(1.0,math.sqrt(case['prompt_tokens']/max(1,prompt_length)))
                        expected_tokens=cap_tokens;unmeasured+=1
                    if case['dataset']!='gsm8k' or name!=base or case['budget']!=193 or case['temperature']!=1:
                        unmeasured+=1
                    projected+=expected_tokens/rate*case['repeats']*factor
                    cap_projection+=cap_tokens/rate*case['repeats']*factor
                    calls+=math.ceil(case['requests']/c)*case['repeats']
            stages[phase]={'eos_limited_projection_hours':projected/3600,
                'all_requests_reach_cap_projection_hours':cap_projection/3600,
                'planning_range_hours':[projected/3600*0.7,cap_projection/3600*1.8],
                'method_batch_executions':calls,'C8_interpolated_shape_count':interpolated,
                'unmeasured_condition_count':unmeasured}
        by_model[model]={'stages':stages,
            'eos_limited_hours':sum(s['eos_limited_projection_hours'] for s in stages.values()),
            'cap_projection_hours':sum(s['all_requests_reach_cap_projection_hours'] for s in stages.values()),
            'planning_range_hours':[sum(s['planning_range_hours'][i] for s in stages.values()) for i in [0,1]]}
    single_range=[sum(m['planning_range_hours'][i] for m in by_model.values()) for i in [0,1]]
    twin_range=[max(m['planning_range_hours'][i] for m in by_model.values()) for i in [0,1]]
    return {'scope':'implemented registered batch suite only, not all conference prerequisites',
        'basis':'actual outer wall seconds and outputs from both complete T1/R193 GSM8K pilots',
        'measured_pilot_execution_seconds':measured,'coverage':coverage,'by_model':by_model,
        'single_H20_planning_range_hours':single_range,'two_matched_H20_model_sharded_range_hours':twin_range,
        'single_H20_eos_limited_projection_hours':sum(m['eos_limited_hours'] for m in by_model.values()),
        'single_H20_cap_projection_hours':sum(m['cap_projection_hours'] for m in by_model.values()),
        'uncertainty':'0.7x EOS-limited to1.8x cap projection is an explicit engineering planning allowance, NOT a confidence interval or hard bound. C8 uses log-rate interpolation; other tasks, budgets, controls and long contexts not measured in this pilot.',
        'excluded':'official baseline/native serving integration and runs, oracle/range confirmation, full code scoring and new data/weight downloads; measured model load/warmup and deployment overhead should be added',
        'additional_server_purchase':'not performed; needs user decision and new host credentials'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args();report=estimate(args.directory)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
