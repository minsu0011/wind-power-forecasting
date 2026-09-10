"""Third append-only recovery: reuse 9865..13037, fetch 13038..13149."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
PARENT=PROJECT/"scripts/download_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery2.py"
ROOT=PROJECT/"artifacts/final_submission_sprint_20260812/kma_d1_1100_posthoc_probe_v1/wsd_typ01"
CONFIG=PROJECT/"configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery3_execution.json"
PREGATE=PROJECT/"artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01/PREGATE_2022_2024_SUMMARY_V8.json"
SUMMARY=ROOT/"YEAR2025_SUMMARY_POSTHOC_V1_RECOVERY3.json"
REUSED_MAX,REUSED_COUNT,FIRST_NETWORK,REMAINING=13037,3173,13038,112
EXTRA_ATTEMPTS,EXTRA_BYTES,MAX_CALLS,LIMIT=6,2_047_782,13_155,4_700_000_000

def load_parent():
 s=importlib.util.spec_from_file_location("_posthoc_recovery3_parent",PARENT)
 if s is None or s.loader is None: raise RuntimeError("cannot load recovery2 parent")
 m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
parent=load_parent();base=parent.base;base.OUTPUT=ROOT

def ident(p): return {"path":p.relative_to(PROJECT).as_posix(),"bytes":p.stat().st_size,"sha256":base.sha256_file(p)}
def verify_config():
 c=json.loads(CONFIG.read_text(encoding="ascii"))
 if c.get("schema_version")!=1 or c.get("status")!="FROZEN_BEFORE_SINGLE_THIRD_RECOVERY_NETWORK_EXECUTION" or c.get("script_identity")!=ident(Path(__file__).resolve()): raise RuntimeError("recovery3 config mismatch")
 for d in c.get("bound_inputs",[]):
  p=PROJECT/d["path"]
  if ident(p)!=d: raise RuntimeError(f"recovery3 input drift: {p.name}")
 if c.get("actual_argv")!=[".venv/Scripts/python.exe","-B","scripts/download_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery3.py","--workers","1"]: raise RuntimeError("recovery3 argv mismatch")
 return c
def prior():
 s=json.loads(PREGATE.read_text(encoding="ascii"))
 if s.get("records")!=9864 or s.get("sequence_max")!=9864: raise RuntimeError("pregate drift")
 return int(s["response_bytes_sum"]),int(s["response_bytes_max"])
def validate_prefix(specs):
 prefix=[x for x in specs if 9865<=x.sequence<=REUSED_MAX]
 if len(prefix)!=REUSED_COUNT: raise RuntimeError("prefix census mismatch")
 for x in prefix:
  if base.read_existing(x) is None: raise RuntimeError(f"prefix missing {x.sequence}")
 first=next(x for x in specs if x.sequence==FIRST_NETWORK);raw,meta=base.paths_for(first)
 if raw.exists() or meta.exists(): raise RuntimeError("13038 unexpectedly persisted; amend ledger")
def run(specs):
 ps,_=prior();key=base.user_key();out=[]
 for x in specs:
  out.append(base.acquire_one(x,key))
  if ps+EXTRA_BYTES+sum(int(y["response_bytes"]) for y in out)>LIMIT: raise RuntimeError("4.7GB stop")
  if len(out)%10==0 or len(out)==len(specs): print(f"progress={len(out)}/{len(specs)} latest_sequence={x.sequence}",flush=True)
 return out
def summarize(r):
 ps,pm=prior();sizes=[int(x["response_bytes"]) for x in r];bad=[int(x["sequence"]) for x in r if not bool(x.get("selected_cells_all_valid_0_75",True))];elapsed=[float(x["elapsed_seconds"]) for x in r if not x.get("reused_existing")];mx=max([pm,*sizes]);projected=mx*MAX_CALLS
 return {"schema_version":1,"experiment":"KMA_POSTHOC_HIGH_RISK_RECOVERY3","mode":"year2025","records":len(r),"sequence_min":min(int(x["sequence"]) for x in r),"sequence_max":max(int(x["sequence"]) for x in r),"persisted_prefix_reused_count":REUSED_COUNT,"persisted_prefix_reused_sequence_range":[9865,REUSED_MAX],"first_recovery_network_sequence":FIRST_NETWORK,"remaining_network_calls":REMAINING,"response_bytes_sum":sum(sizes),"response_bytes_max":max(sizes),"prior_successful_calls":9864,"prior_successful_response_bytes":ps,"prior_extra_physical_attempts":EXTRA_ATTEMPTS,"prior_extra_physical_bytes_conservative":EXTRA_BYTES,"maximum_physical_attempts":MAX_CALLS,"total_response_bytes_accounted":ps+EXTRA_BYTES+sum(sizes),"projected_physical_bytes_by_observed_max":projected,"byte_gate_limit":LIMIT,"byte_gate_pass":projected<=LIMIT and ps+EXTRA_BYTES+sum(sizes)<=LIMIT,"mean_request_elapsed_seconds_new_only":float(base.np.mean(elapsed)) if elapsed else 0.0,"invalid_selected_anchor_count":len(bad),"invalid_selected_anchor_sequences":bad,"posthoc_high_risk":True,"independent_confirmation":False,"secret_persisted":False}
def main():
 a=argparse.ArgumentParser();a.add_argument("--workers",type=int,default=1);a.add_argument("--verify-only",action="store_true");z=a.parse_args()
 if z.workers!=1: raise ValueError("workers must be 1")
 verify_config();prior();specs=[x for x in base.all_specs() if x.operating_year==2025];validate_prefix(specs)
 if z.verify_only: print(json.dumps({"verify_only":"PASS","reused":REUSED_COUNT,"first_network_sequence":FIRST_NETWORK,"remaining_network_calls":REMAINING},sort_keys=True));return
 r=run(specs);s=summarize(r)
 if s["records"]!=3285 or s["sequence_min"]!=9865 or s["sequence_max"]!=13149: raise RuntimeError("completion census mismatch")
 base.write_exclusive(SUMMARY,base.canonical_json_bytes(s))
 if not s["byte_gate_pass"]: raise RuntimeError("byte gate failed")
 print(json.dumps(s,sort_keys=True),flush=True)
if __name__=="__main__": main()
