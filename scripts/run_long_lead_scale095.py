from __future__ import annotations

import atexit
import csv
import hashlib
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from scripts.run_ecmwf_ifs025_paired_increment import HEAVY_GUARD_PATH,atomic_json,atomic_parquet,file_record,pid_is_alive,sha256
from scripts.run_run_sequence_residual import _read_stage1_baseline
from src.metric import CAPACITY_KWH,TARGET_COLS,group_metrics,score_details


CONFIG=ROOT/"configs/long_lead_scale095_preregister_v1.json"; SIDECAR=CONFIG.with_suffix(".sha256")
LABEL=Path(r"data/local/open/train/train_labels.csv")
ART=ROOT/"artifacts"; OUT=ART/"postgate/long_lead_scale095_strict_v1"
BASE24={"primary_corrected_v3":ART/"oof/gate2024_locked_v3_cf_fix.parquet","interaction_recent_v4":ART/"oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"}
Y23=pd.date_range("2023-01-01 01:00","2024-01-01 00:00",freq="h",name="forecast_kst_dtm")
Y24=pd.date_range("2024-01-01 01:00","2025-01-01 00:00",freq="h",name="forecast_kst_dtm")


def year_slices(year:int)->dict[str,pd.DatetimeIndex]:
    start=pd.Timestamp(f"{year}-01-01 01:00"); end=pd.Timestamp(f"{year+1}-01-01 00:00")
    return {
      "full":pd.date_range(start,end,freq="h"),
      "H1":pd.date_range(start,pd.Timestamp(f"{year}-07-01 00:00"),freq="h"),
      "H2":pd.date_range(pd.Timestamp(f"{year}-07-01 01:00"),end,freq="h"),
      "Q1":pd.date_range(start,pd.Timestamp(f"{year}-04-01 00:00"),freq="h"),
      "Q2":pd.date_range(pd.Timestamp(f"{year}-04-01 01:00"),pd.Timestamp(f"{year}-07-01 00:00"),freq="h"),
      "Q3":pd.date_range(pd.Timestamp(f"{year}-07-01 01:00"),pd.Timestamp(f"{year}-10-01 00:00"),freq="h"),
      "Q4":pd.date_range(pd.Timestamp(f"{year}-10-01 01:00"),end,freq="h"),
    }


def lead_values(index:pd.DatetimeIndex)->np.ndarray:
    position=np.where(index.hour.to_numpy()==0,24,index.hour.to_numpy())
    values=11+position
    if not np.array_equal(np.sort(np.unique(values)),np.arange(12,36)): raise ValueError("lead support differs")
    return values


def long_lead_mask(index:pd.DatetimeIndex)->np.ndarray:
    mask=lead_values(index)>=24
    key=(index-pd.Timedelta(hours=1)).normalize()
    counts=pd.Series(mask.astype(np.int8),index=index).groupby(key).sum()
    if not counts.eq(12).all(): raise ValueError("long-lead mask does not contain 12 rows/run")
    return mask


def candidate_from_baseline(baseline:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    mask=long_lead_mask(baseline.index); candidate=baseline.copy(); candidate.loc[mask,:]=candidate.loc[mask,:]*0.95
    audit=pd.DataFrame({"lead":lead_values(baseline.index),"long_lead":mask,"factor":np.where(mask,.95,1.)},index=baseline.index)
    return candidate,audit


def acquire(path:Path)->dict[str,Any]:
    path.parent.mkdir(parents=True,exist_ok=True); owner={"pid":os.getpid(),"token":secrets.token_hex(16),"experiment_id":"long_lead_scale095_strict_v1","created_utc":datetime.now(timezone.utc).isoformat()}
    for _ in range(2):
        try: fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            current=json.loads(path.read_text(encoding="utf-8"))
            if pid_is_alive(int(current["pid"])): raise RuntimeError(f"guard held: {current}")
            path.unlink(); continue
        with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(owner,f,sort_keys=True); f.write("\n")
        return owner
    raise RuntimeError("guard acquisition failed")


def release(path:Path,owner:Mapping[str,Any])->None:
    try: current=json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError,OSError,json.JSONDecodeError): return
    if current.get("pid")==owner.get("pid") and current.get("token")==owner.get("token"): path.unlink()


def verify()->dict[str,Any]:
    if sha256(CONFIG)!=SIDECAR.read_text().split()[0]: raise RuntimeError("prereg mismatch")
    p=json.loads(CONFIG.read_text(encoding="utf-8")); expected={
      ART/"oof/dev2023_locked_v3.parquet":p["stage1"]["baseline_g12"]["sha256"],
      ART/"oof/g3dev2023h2_candidates.parquet":p["stage1"]["baseline_g3_components"]["sha256"],
      ART/"postgate/shared_q07_multiseed_strict/oof/stage1_corrected_v3_baseline.parquet":p["stage1"]["exact_baseline_reference"]["sha256"],
      ROOT/"src/metric.py":p["metric"]["sha256"],
    }
    bad={str(path):(want,sha256(path)) for path,want in expected.items() if sha256(path)!=want}
    if bad: raise RuntimeError(f"input hash mismatch {bad}")
    return p


def verify_stage2_inputs(config: Mapping[str, Any]) -> None:
    expected={path:config["stage2_only_after_stage1_pass"]["baselines"][i]["sha256"] for i,path in enumerate(BASE24.values())}
    bad={str(path):(want,sha256(path)) for path,want in expected.items() if sha256(path)!=want}
    if bad: raise RuntimeError(f"Stage2 input hash mismatch {bad}")


def source_lock(out:Path)->Path:
    files=[CONFIG,SIDECAR,ROOT/"scripts/run_long_lead_scale095.py",ROOT/"tests/test_long_lead_scale095.py",ROOT/"scripts/run_run_sequence_residual.py",ROOT/"src/lead_time_experts.py",ROOT/"src/metric.py"]
    path=out/"source_lock_before_candidate_or_labels.json"; atomic_json({"status":"locked_before_baseline_candidate_or_label_read","sources":[file_record(p) for p in files],"public_or_scale_artifact_files":0,"2025_access":0},path); return path


class Labels:
    def __init__(self):
        self.f=LABEL.open("rb",buffering=0); self.digest=hashlib.sha256(); header=self.f.readline(); self.digest.update(header); self.rows=0
        if next(csv.reader([header.decode("utf-8-sig").strip()])) != ["kst_dtm",*TARGET_COLS]: raise ValueError("header differs")
    def read23(self)->tuple[pd.DataFrame,dict[str,Any]]:
        times=[]; values=[]
        while self.rows<17520:
            line=self.f.readline(); self.digest.update(line); self.rows+=1; ts=pd.Timestamp(line.split(b",",1)[0].decode("ascii"))
            if ts>=Y23[0]:
                fields=next(csv.reader([line.decode("utf-8").strip()])); times.append(ts); values.append([float(v) if v else np.nan for v in fields[1:]])
        frame=pd.DataFrame(values,index=pd.DatetimeIndex(times,name="forecast_kst_dtm"),columns=TARGET_COLS)
        if not frame.index.equals(Y23): raise ValueError("2023 label index differs")
        return frame,self.record("2023")
    def read24(self)->tuple[pd.DataFrame,dict[str,Any]]:
        times=[]; values=[]
        for _ in range(len(Y24)):
            line=self.f.readline(); self.digest.update(line); self.rows+=1; fields=next(csv.reader([line.decode("utf-8").strip()])); times.append(pd.Timestamp(fields[0])); values.append([float(v) if v else np.nan for v in fields[1:]])
        frame=pd.DataFrame(values,index=pd.DatetimeIndex(times,name="forecast_kst_dtm"),columns=TARGET_COLS)
        if not frame.index.equals(Y24): raise ValueError("2024 label index differs")
        return frame,self.record("2024")
    def record(self,phase:str)->dict[str,Any]: return {"phase":phase,"rows":self.rows,"bytes":self.f.tell(),"prefix_sha256":self.digest.hexdigest()}
    def close(self): self.f.close()


def gm(actual,pred,group)->dict[str,float]:
    x=group_metrics(actual,pred,CAPACITY_KWH[group],group_name=group).as_dict(); x["total_score"]=.5*(x["one_minus_nmae"]+x["ficr"]); return x
def comp(actual,base,cand,group)->dict[str,Any]:
    b=gm(actual,base,group); c=gm(actual,cand,group); return {"baseline":b,"candidate":c,"delta":{k:c[k]-b[k] for k in ("total_score","one_minus_nmae","ficr")}}


def main()->None:
    cfg=verify()
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); closure=source_lock(OUT); owner=acquire(HEAVY_GUARD_PATH); atexit.register(release,HEAVY_GUARD_PATH,owner)
    baseline,evidence=_read_stage1_baseline(ART); candidate,mask=candidate_from_baseline(baseline)
    cpath=OUT/"predictions/stage1_candidate.parquet"; mpath=OUT/"predictions/stage1_lead_mask.parquet"; atomic_parquet(candidate,cpath); atomic_parquet(mask,mpath)
    lock=OUT/"stage1_candidate_lock_before_labels.json"; atomic_json({"candidate":file_record(cpath),"lead_mask":file_record(mpath),"baseline_evidence":evidence,"candidate_value_sha256":hashlib.sha256(candidate.to_numpy(dtype=np.float64).tobytes()).hexdigest(),"mask_bytes_sha256":hashlib.sha256(mask.to_numpy().tobytes()).hexdigest(),"label_bytes_read":0},lock)
    cursor=Labels(); labels23,access23=cursor.read23(); atomic_json({"candidate_lock":file_record(lock),"access":access23},OUT/"stage1_label_access.json")
    slices=year_slices(2023); registered={"kpx_group_1":["full","H1","H2","Q1","Q2","Q3","Q4"],"kpx_group_2":["full","H1","H2","Q1","Q2","Q3","Q4"],"kpx_group_3":["H2","Q3","Q4"]}
    comparisons={}; passed=True
    for group,names in registered.items():
        comparisons[group]={}
        for name in names:
            x=comp(labels23.loc[slices[name],group],baseline.loc[slices[name],group],candidate.loc[slices[name],group],group); comparisons[group][name]=x; passed &= x["delta"]["total_score"]>0
        full=comparisons[group][names[0]]["delta"]; passed &= full["one_minus_nmae"]>=0 and full["ficr"]>=0
    result={"status":"stage1_promoted" if passed else "rejected_on_stage1","passed":bool(passed),"comparisons":comparisons,"selection_unsafe":True,"public_scale_artifact_files":0,"2024_label_candidate_metric":0,"2025_access":0}
    rpath=OUT/"stage1_results.json"; atomic_json(result,rpath); atomic_json({"status":result["status"],"results":file_record(rpath),"stage2_authorized":bool(passed)},OUT/("stage1_promotion_lock.json" if passed else "rejection.json"))
    if passed:
        verify_stage2_inputs(cfg)
        labels24_candidates={}
        for name,path in BASE24.items():
            b=pd.read_parquet(path).loc[Y24,list(TARGET_COLS)].astype(np.float64); c,a=candidate_from_baseline(b); labels24_candidates[name]=(b,c); atomic_parquet(c,OUT/f"predictions/stage2_{name}.parquet")
        s2lock=OUT/"stage2_candidates_lock_before_labels.json"; atomic_json({"candidates":{n:file_record(OUT/f"predictions/stage2_{n}.parquet") for n in labels24_candidates},"label_cells_read":0},s2lock)
        labels24,access24=cursor.read24(); atomic_json({"candidate_lock":file_record(s2lock),"access":access24},OUT/"stage2_label_access.json")
        s2comp={}; pass2=True; slices24=year_slices(2024)
        for name,(b,c) in labels24_candidates.items():
            s2comp[name]={"groups":{},"mixed":{}}
            for group in TARGET_COLS:
                s2comp[name]["groups"][group]={}
                for sn,idx in slices24.items():
                    x=comp(labels24.loc[idx,group],b.loc[idx,group],c.loc[idx,group],group); s2comp[name]["groups"][group][sn]=x; pass2 &= x["delta"]["total_score"]>0
                full=s2comp[name]["groups"][group]["full"]["delta"]; pass2 &= full["one_minus_nmae"]>=0 and full["ficr"]>=0
            for sn,idx in slices24.items():
                bm=score_details(labels24.loc[idx],b.loc[idx]).as_dict(); cm=score_details(labels24.loc[idx],c.loc[idx]).as_dict(); d={k:cm[k]-bm[k] for k in ("total_score","one_minus_nmae","ficr")}; s2comp[name]["mixed"][sn]={"baseline":bm,"candidate":cm,"delta":d}; pass2 &= d["total_score"]>0
            full=s2comp[name]["mixed"]["full"]["delta"]; pass2 &= full["one_minus_nmae"]>=0 and full["ficr"]>=0
        s2={"status":"promoted_for_separate_lock_only" if pass2 else "rejected_on_stage2","passed":bool(pass2),"comparisons":s2comp,"2025_access":0}; atomic_json(s2,OUT/"stage2_results.json"); atomic_json({"status":s2["status"],"results":file_record(OUT/"stage2_results.json")},OUT/("promotion_lock.json" if pass2 else "stage2_rejection.json"))
    cursor.close(); files=[p for p in OUT.rglob("*") if p.is_file()]; manifest=OUT/"manifest.json"; atomic_json({"experiment_id":cfg["experiment_id"],"preregister":file_record(CONFIG),"outputs_excluding_manifest":[file_record(p) for p in files],"stage1_passed":bool(passed),"selection_unsafe":True,"public_scale_artifact_files":0,"2025_access":0,"csv":False},manifest); (OUT/"manifest.sha256").write_text(f"{sha256(manifest)}  manifest.json\n",encoding="ascii")
    print(f"stage1_passed={passed} results={sha256(rpath)} manifest={sha256(manifest)}",flush=True); release(HEAVY_GUARD_PATH,owner)


if __name__=="__main__": main()
