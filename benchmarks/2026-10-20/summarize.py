#!/usr/bin/env python3
"""Summarize archived timing and accuracy reports using identical scene sets.

Run without arguments to regenerate summary-final.json beside this script.
--results, --baseline, and --output accept alternate JSON or gzip report paths.
Only the Python standard library is required.
"""
import argparse
import gzip
import json
import math
from pathlib import Path

archive = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=archive / "domain-final.json.gz")
parser.add_argument("--baseline", type=Path, default=archive / "baseline-primary.json.gz")
parser.add_argument("--output", type=Path, default=archive / "summary-final.json")
args = parser.parse_args()

def read_report(path):
    """Read a plain or gzip-compressed benchmark report."""
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)

final = read_report(args.results)
base = read_report(args.baseline)
variants={v['name']:v for v in final['variants']}
def stats(values):
 values=sorted(values)
 return {k:values[min(len(values)-1,max(0,math.ceil(len(values)*p)-1))] for k,p in [('median',.5),('p95',.95),('max',1)]} if values else None
def samples(scenes,api):return [t for s in scenes for t in s[api+'_samples_ms']]
def accuracy(scenes):
 return {'scenes':len(scenes),'passed':sum(s['accuracy']['passed'] for s in scenes),'expected_cards':sum(s['accuracy']['expected_count'] for s in scenes),'detected_cards':sum(s['accuracy']['detected_count'] for s in scenes),
 'failures':[{'id':s['id'],'expected_cards':s['accuracy']['expected_count'],'detected_cards':s['accuracy']['detected_count'],
 'max_corner_error':max((m['max_corner_error'] for m in s['accuracy']['matches']),default=None),
 'max_pixel_mae':max((m['interior_rgb_mae'] for m in s['accuracy']['matches'] if m['interior_rgb_mae'] is not None),default=None)} for s in scenes if not s['accuracy']['passed']]}
def compare(before,after,domain,description):
 indices=[i for i,s in enumerate(after) if domain(s)]
 left=[before[i] for i in indices];right=[after[i] for i in indices]
 assert [s['id'] for s in left]==[s['id'] for s in right], 'scene sets/order must match'
 return {'description':description,'scene_ids':[s['id'] for s in right],'before_accuracy':accuracy(left),'after_accuracy':accuracy(right),
  **{api:{'before_ms':stats(samples(left,api)),'after_ms':stats(samples(right,api)),
  'median_speedup':stats(samples(left,api))['median']/stats(samples(right,api))['median']} for api in ['detect','extract']}}
result={'metadata':final['metadata'],'methodology':final['methodology'],'variants':[], 'comparisons':[]}
for v in final['variants']:
 eligible=[s for s in v['scenes'] if s['domain_eligible']]
 result['variants'].append({'name':v['name'],'settings':v['settings'],'eligible_summary':v['eligible_summary'],'accuracy':accuracy(eligible)})
for before,after in [('full-640','full-640-roi'),('full-640','full-640-angle'),('full-640-roi','full-640-roi-angle'),('fast-480','fast-480-roi'),('fast-480','fast-480-angle'),('fast-480-roi','fast-480-roi-angle')]:
 result['comparisons'].append(compare(variants[before]['scenes'],variants[after]['scenes'],lambda s:s['domain_eligible'],f'{before} -> {after}, identical eligible scenes'))
primary=[s for s in variants['default']['scenes'] if Path(s['manifest']).parent.name=='scenes']
result['baseline_comparison']=compare(base['variants'][0]['scenes'],primary,lambda s:True,'frozen pre-change Full -> optimized Full, same68primary regressions')
result['baseline_metadata']=base['metadata']
args.output.write_text(json.dumps(result,indent=2)+'\n')
for v in result['variants']:
 s=v['eligible_summary'];print(v['name'],f"{s['passed']}/{s['scenes']}",'detect',s['detect_ms'],'extract',s['extract_ms'])
for p in result['comparisons']+[result['baseline_comparison']]:
 print(p['description'],p['before_accuracy']['scenes'],'scenes; extraction',p['extract'])
