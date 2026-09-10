"""Distinct frozen finalizer for recovery3/V5 KMA posthoc rank probe."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/"scripts/build_kma_d1_1100_wsd_posthoc_rank_probe_v1.py"
CONFIG=PROJECT/"configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_finalizer_recovery3_execution_v4.json"
INCIDENT=PROJECT/"configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_finalizer_mutation_incident.json"
ROOT=PROJECT/"artifacts/final_submission_sprint_20260812/kma_d1_1100_posthoc_probe_v1"
WSD=ROOT/"wsd_typ01"
KMA=WSD/"materialized_recovery3_v5/KMA_D1_1100_WSD_HOURLY_2025_POSTHOC_V1_RECOVERY3_V5.parquet"
MANIFEST=WSD/"materialized_recovery3_v5/MANIFEST_2025_POSTHOC_V1_RECOVERY3_V5.json"
SUMMARY=WSD/"YEAR2025_SUMMARY_POSTHOC_V1_RECOVERY3.json"
MAT_CONFIG=PROJECT/"configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_recovery3_execution_v5.json"
MAT_SCRIPT=PROJECT/"scripts/materialize_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery3_v5.py"
PRE=PROJECT/"artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01/PREGATE_2022_2024_SUMMARY_V8.json"

def load_source():
 s=importlib.util.spec_from_file_location("_posthoc_finalizer_recovery3_v4_source",SOURCE)
 if s is None or s.loader is None:raise RuntimeError("cannot load implementation source")
 m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
base=load_source()
base.FINALIZER_CONFIG=CONFIG;base.INCIDENT=INCIDENT;base.MATERIALIZER_CONFIG=MAT_CONFIG;base.MATERIALIZER=MAT_SCRIPT;base.KMA_TEST=KMA;base.KMA_TEST_MANIFEST=MANIFEST;base.SUMMARY_2025=SUMMARY;base.PREGATE_SUMMARY=PRE
base.MODEL_ROOT=ROOT/"final_model_recovery3_v4";base.FINAL_CSV=ROOT/"04_HIGH_RISK_POSTHOC_KMA_R2_VERTICAL_RANK_PROBE_RECOVERY3_V4.csv";base.FINAL_MANIFEST=ROOT/"04_HIGH_RISK_POSTHOC_KMA_R2_VERTICAL_RANK_PROBE_RECOVERY3_V4.manifest.json"

def relid(p):return {"path":p.relative_to(PROJECT).as_posix(),"bytes":p.stat().st_size,"sha256":base.sha256_file(p)}
def verify_execution():
 c=json.loads(CONFIG.read_text(encoding="ascii"))
 if c.get("schema_version")!=4 or c.get("status")!="FROZEN_BEFORE_DISTINCT_RECOVERY3_V4_FINAL_REFIT_OR_CSV" or c.get("script_identity")!=relid(Path(__file__).resolve()):raise RuntimeError("distinct finalizer V4 config mismatch")
 verified=[]
 for d in c.get("bound_inputs",[]):
  q=Path(d["path"]);p=q if q.is_absolute() else PROJECT/q;a=base.identity(p,relative=not q.is_absolute())
  if a!=d:raise RuntimeError(f"distinct finalizer input drift: {p.name}")
  verified.append(a)
 if c.get("actual_argv")!=[".venv/Scripts/python.exe","-B","scripts/build_kma_d1_1100_wsd_posthoc_rank_probe_recovery3_v4.py"]:raise RuntimeError("distinct finalizer argv mismatch")
 return base.identity(CONFIG),verified

def verify_post_sources():
 s=json.loads(SUMMARY.read_text(encoding="ascii"));pre=json.loads(PRE.read_text(encoding="ascii"));mx=int(s["response_bytes_max"]);total=int(pre["response_bytes_sum"])+2047782+int(s["response_bytes_sum"])
 e={"records":3285,"sequence_min":9865,"sequence_max":13149,"persisted_prefix_reused_count":3173,"first_recovery_network_sequence":13038,"remaining_network_calls":112,"prior_extra_physical_attempts":6,"prior_extra_physical_bytes_conservative":2047782,"maximum_physical_attempts":13155}
 if any(s.get(k)!=v for k,v in e.items()) or s.get("byte_gate_pass") is not True or int(s["projected_physical_bytes_by_observed_max"])!=mx*13155 or int(s["total_response_bytes_accounted"])!=total:raise RuntimeError("distinct finalizer recovery summary mismatch")
 m=json.loads(MANIFEST.read_text(encoding="ascii"));ser=json.dumps(m,sort_keys=True)
 if m.get("rows")!=8760 or m.get("parquet")!=base.identity(KMA,relative=True) or m.get("materializer")!=base.identity(MAT_SCRIPT,relative=True) or m.get("secret_persisted") is not False:raise RuntimeError("V5 materializer manifest mismatch")
 for p in (SUMMARY,MAT_CONFIG,MAT_SCRIPT,INCIDENT):
  if base.sha256_file(p) not in ser and p in (SUMMARY,MAT_CONFIG,MAT_SCRIPT):raise RuntimeError(f"V5 lineage missing {p.name}")
 return {"summary":base.identity(SUMMARY),"parquet":base.identity(KMA),"manifest":base.identity(MANIFEST),"materializer":base.identity(MAT_SCRIPT),"materializer_config":base.identity(MAT_CONFIG)}

def main():
 import argparse
 a=argparse.ArgumentParser();a.add_argument("--verify-only",action="store_true");z=a.parse_args();execution,bound=verify_execution()
 if z.verify_only:print(json.dumps({"verify_only":"PASS","execution":execution,"bound_inputs":len(bound)},sort_keys=True));return
 # Replace source validators only; exact registered science/model/composition functions remain unchanged.
 base.verify_execution=lambda:execution
 base.verify_post_2025_sources=verify_post_sources
 base.main()
if __name__=="__main__":main()
