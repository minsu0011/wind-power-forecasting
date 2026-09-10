"""Full-provenance offline materializer for the third acquisition recovery."""
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
PARENT=PROJECT/"scripts/materialize_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery2_v3.py"
ROOT=PROJECT/"artifacts/final_submission_sprint_20260812/kma_d1_1100_posthoc_probe_v1/wsd_typ01"
OUTPUT=ROOT/"materialized_recovery3_v4"
SUMMARY=ROOT/"YEAR2025_SUMMARY_POSTHOC_V1_RECOVERY3.json"
PREGATE=PROJECT/"artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01/PREGATE_2022_2024_SUMMARY_V8.json"
CONFIG=PROJECT/"configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_recovery3_execution_v4.json"

def load_parent():
 s=importlib.util.spec_from_file_location("_posthoc_materializer_recovery3_v4_parent",PARENT)
 if s is None or s.loader is None: raise RuntimeError("cannot load V3 parent")
 m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
parent=load_parent();base=parent.base;base.__file__=str(Path(__file__).resolve());base.OUTPUT_ROOT=OUTPUT;base.MATERIALIZER_CONFIG=CONFIG;base.MODE["2025"].update(summary=SUMMARY.name,output="KMA_D1_1100_WSD_HOURLY_2025_POSTHOC_V1_RECOVERY3_V4.parquet",manifest="MANIFEST_2025_POSTHOC_V1_RECOVERY3_V4.json")

def validate_chain():
 c=base.read_json(CONFIG)
 if c.get("schema_version")!=4 or c.get("status")!="FROZEN_BEFORE_POSTHOC_RECOVERY3_V4_MATERIALIZATION" or c.get("materializer")!=base.identity(Path(__file__).resolve()): raise RuntimeError("V4 config mismatch")
 declared=set();verified=[base.identity(CONFIG)]
 for d in c.get("bound_inputs",[]):
  p=PROJECT/d["path"];a=base.identity(p)
  if a!=d: raise RuntimeError(f"V4 input drift: {p.name}")
  declared.add(p);verified.append(a)
 required={PROJECT/x for x in c.get("required_exact_path_census",[])}
 if declared!=required: raise RuntimeError("V4 exact full-provenance path census mismatch")
 return verified
base.validate_chain=validate_chain

def validate_summaries(mode,spec):
 if mode!="2025": raise RuntimeError("V4 authorizes 2025 only")
 s=base.read_json(SUMMARY);expected={"mode":"year2025","records":3285,"sequence_min":9865,"sequence_max":13149,"persisted_prefix_reused_count":3173,"persisted_prefix_reused_sequence_range":[9865,13037],"first_recovery_network_sequence":13038,"remaining_network_calls":112,"prior_successful_calls":9864,"prior_extra_physical_attempts":6,"prior_extra_physical_bytes_conservative":2047782,"maximum_physical_attempts":13155}
 if any(s.get(k)!=v for k,v in expected.items()) or s.get("byte_gate_pass") is not True: raise RuntimeError("V4 recovery3 summary mismatch")
 mx=int(s.get("response_bytes_max",-1));projected=mx*13155
 if int(s.get("projected_physical_bytes_by_observed_max",-1))!=projected or projected>4_700_000_000: raise RuntimeError("V4 projection equation mismatch")
 pre=base.read_json(PREGATE);total=int(pre.get("response_bytes_sum",-1))+2_047_782+int(s.get("response_bytes_sum",-1))
 if int(s.get("total_response_bytes_accounted",-1))!=total or total>4_700_000_000: raise RuntimeError("V4 total equation mismatch")
 bad=s.get("invalid_selected_anchor_sequences")
 if not isinstance(bad,list) or int(s.get("invalid_selected_anchor_count",-1))!=len(bad): raise RuntimeError("V4 invalid census mismatch")
 return [base.identity(SUMMARY),base.identity(PREGATE),base.identity(CONFIG)],s
base.validate_summaries=validate_summaries

_pm=parent.materialize
def materialize(mode):
 t,e=_pm(mode);e["recovery3_v4"]={"summary":base.identity(SUMMARY),"execution_chain":base.identity(CONFIG),"maximum_physical_attempts":13155,"prior_extra_attempts":6,"prior_extra_bytes":2047782,"full_provenance":True};return t,e
base.materialize=materialize
def main():
 a=argparse.ArgumentParser();a.add_argument("--years",choices=("2025",),default="2025");a.add_argument("--verify-only",action="store_true");z=a.parse_args();validate_chain()
 if z.verify_only: print(base.json.dumps({"verify_only":"PASS","mode":"2025","recovery3_v4":True},sort_keys=True));return
 t,e=materialize("2025");o,m=base.publish("2025",t,e);print(base.json.dumps({"rows":t.num_rows,"parquet":o.relative_to(PROJECT).as_posix(),"manifest":m.relative_to(PROJECT).as_posix(),"recovery3_v4":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
