"""V5 offline materializer: call frozen core directly, audit full V4 chain."""
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
V4=PROJECT/"scripts/materialize_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery3_v4.py"
V4_CONFIG=PROJECT/"configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_recovery3_execution_v4.json"
INCIDENT=PROJECT/"configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_v4_attempt_incident.json"
CONFIG=PROJECT/"configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_recovery3_execution_v5.json"
ROOT=PROJECT/"artifacts/final_submission_sprint_20260812/kma_d1_1100_posthoc_probe_v1/wsd_typ01"
SUMMARY=ROOT/"YEAR2025_SUMMARY_POSTHOC_V1_RECOVERY3.json"
PREGATE=PROJECT/"artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01/PREGATE_2022_2024_SUMMARY_V8.json"
OUTPUT=ROOT/"materialized_recovery3_v5"
def load_v4():
 s=importlib.util.spec_from_file_location("_posthoc_materializer_v5_v4",V4)
 if s is None or s.loader is None:raise RuntimeError("cannot load V4")
 m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
v4=load_v4();base=v4.base;base.__file__=str(Path(__file__).resolve());base.OUTPUT_ROOT=OUTPUT;base.MATERIALIZER_CONFIG=CONFIG;base.MODE["2025"].update(summary=SUMMARY.name,output="KMA_D1_1100_WSD_HOURLY_2025_POSTHOC_V1_RECOVERY3_V5.parquet",manifest="MANIFEST_2025_POSTHOC_V1_RECOVERY3_V5.json")
core=v4.parent.parent.parent.wrapper._base_materialize
def validate_chain():
 c=base.read_json(CONFIG)
 if c.get("schema_version")!=5 or c.get("status")!="FROZEN_BEFORE_POSTHOC_RECOVERY3_V5_MATERIALIZATION" or c.get("materializer")!=base.identity(Path(__file__).resolve()):raise RuntimeError("V5 config mismatch")
 verified=[base.identity(CONFIG)]
 for d in c.get("bound_inputs",[]):
  p=PROJECT/d["path"];a=base.identity(p)
  if a!=d:raise RuntimeError(f"V5 input drift: {p.name}")
  verified.append(a)
 vc=base.read_json(V4_CONFIG)
 if vc.get("materializer")!=base.identity(V4):raise RuntimeError("V4 binding drift")
 for d in vc.get("bound_inputs",[]):
  p=PROJECT/d["path"]
  if base.identity(p)!=d:raise RuntimeError(f"V4 transitive input drift: {p.name}")
 if {PROJECT/x for x in vc["required_exact_path_census"]}!={PROJECT/d["path"] for d in vc["bound_inputs"]}:raise RuntimeError("V4 full path census drift")
 return verified+[base.identity(V4_CONFIG),base.identity(V4),base.identity(INCIDENT)]
base.validate_chain=validate_chain
def validate_summaries(mode,spec):
 if mode!="2025":raise RuntimeError("V5 2025 only")
 s=base.read_json(SUMMARY);e={"mode":"year2025","records":3285,"sequence_min":9865,"sequence_max":13149,"persisted_prefix_reused_count":3173,"persisted_prefix_reused_sequence_range":[9865,13037],"first_recovery_network_sequence":13038,"remaining_network_calls":112,"prior_extra_physical_attempts":6,"prior_extra_physical_bytes_conservative":2047782,"maximum_physical_attempts":13155}
 if any(s.get(k)!=v for k,v in e.items()) or s.get("byte_gate_pass") is not True:raise RuntimeError("V5 summary mismatch")
 mx=int(s["response_bytes_max"]);pre=base.read_json(PREGATE);total=int(pre["response_bytes_sum"])+2047782+int(s["response_bytes_sum"])
 if int(s["projected_physical_bytes_by_observed_max"])!=mx*13155 or int(s["total_response_bytes_accounted"])!=total or total>4700000000:raise RuntimeError("V5 byte equation mismatch")
 return [base.identity(SUMMARY),base.identity(PREGATE)],s
base.validate_summaries=validate_summaries
def materialize(mode):
 t,e=core(mode);e["v5"]={"full_v4_chain_verified":True,"v4_incident":base.identity(INCIDENT),"summary":base.identity(SUMMARY),"maximum_physical_attempts":13155,"prior_extra_bytes":2047782};return t,e
base.materialize=materialize
def main():
 a=argparse.ArgumentParser();a.add_argument("--years",choices=("2025",),default="2025");a.add_argument("--verify-only",action="store_true");z=a.parse_args();validate_chain()
 if z.verify_only:print(base.json.dumps({"verify_only":"PASS","recovery3_v5":True},sort_keys=True));return
 t,e=materialize("2025");o,m=base.publish("2025",t,e);print(base.json.dumps({"rows":t.num_rows,"parquet":o.relative_to(PROJECT).as_posix(),"manifest":m.relative_to(PROJECT).as_posix(),"recovery3_v5":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
