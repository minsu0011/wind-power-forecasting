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

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_shared_q07_multiseed as shared
from scripts.run_ecmwf_ifs025_paired_increment import (
    HEAVY_GUARD_PATH,
    atomic_json,
    atomic_parquet,
    file_record,
    pid_is_alive,
    sha256,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details


CONFIG = ROOT / "configs/daily_energy_ratio_g12_identity_g3_preregister_v3.json"
SIDECAR = CONFIG.with_suffix(".sha256")
V2_CONFIG = ROOT / "configs/daily_energy_ratio_g12_identity_g3_preregister_v2.json"
V1_CONFIG = ROOT / "configs/daily_energy_ratio_g12_identity_g3_preregister_v1.json"
INCIDENT = ROOT / "configs/daily_energy_ratio_g12_identity_g3_preregister_v2_invalidated.json"
REJECTION = ROOT / "configs/daily_energy_ratio_calibrator_feasibility_rejection_v1.json"
RAW_DIR = Path(r"data/local/open")
LABEL_PATH = RAW_DIR / "train/train_labels.csv"
BASE_2023 = ROOT / "artifacts/oof/dev2023_locked_v3.parquet"
CACHE = {g: ROOT / f"artifacts/cache/{g}_weather_train.parquet" for g in TARGET_COLS[:2]}
BASE_2024 = {
    "primary_corrected_v3": ROOT / "artifacts/oof/gate2024_locked_v3_cf_fix.parquet",
    "interaction_recent_v4": ROOT / "artifacts/oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet",
}
OUT = ROOT / "artifacts/postgate/daily_energy_ratio_g12_identity_g3_strict_v3"
GROUPS = tuple(TARGET_COLS[:2])
WEATHER_COLUMNS = ("ldaps__idw__hub_ws", "gfs__idw__hub_ws", "cross__hub_ws_mean")
FEATURE_COLUMNS = (
    "baseline_daily_mean_cf",
    "ldaps_ws_mean", "ldaps_ws_std", "ldaps_ws_q25", "ldaps_ws_q50", "ldaps_ws_q75",
    "gfs_ws_mean", "gfs_ws_std", "gfs_ws_q25", "gfs_ws_q50", "gfs_ws_q75",
    "ldaps_minus_gfs_mean", "ldaps_minus_gfs_std", "ldaps_minus_gfs_q25", "ldaps_minus_gfs_q50", "ldaps_minus_gfs_q75",
    "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2",
)
Y2023 = pd.date_range("2023-01-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm")
Y2024 = pd.date_range("2024-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")
S1 = {
    "Q1": pd.date_range("2023-01-01 01:00", "2023-04-01 00:00", freq="h", name="forecast_kst_dtm"),
    "Q2": pd.date_range("2023-04-01 01:00", "2023-07-01 00:00", freq="h", name="forecast_kst_dtm"),
    "Q3": pd.date_range("2023-07-01 01:00", "2023-10-01 00:00", freq="h", name="forecast_kst_dtm"),
    "Q4": pd.date_range("2023-10-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm"),
}
S1_SLICES = {
    "Q2": S1["Q2"], "Q3": S1["Q3"], "Q4": S1["Q4"],
    "H2": S1["Q3"].append(S1["Q4"]),
    "Q2_Q4_full": S1["Q2"].append(S1["Q3"]).append(S1["Q4"]),
}
S2_SLICES = {
    "full": Y2024,
    "H1": pd.date_range("2024-01-01 01:00", "2024-07-01 00:00", freq="h"),
    "H2": pd.date_range("2024-07-01 01:00", "2025-01-01 00:00", freq="h"),
    "Q1": pd.date_range("2024-01-01 01:00", "2024-04-01 00:00", freq="h"),
    "Q2": pd.date_range("2024-04-01 01:00", "2024-07-01 00:00", freq="h"),
    "Q3": pd.date_range("2024-07-01 01:00", "2024-10-01 00:00", freq="h"),
    "Q4": pd.date_range("2024-10-01 01:00", "2025-01-01 00:00", freq="h"),
}


def acquire_guard(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"pid": os.getpid(), "token": secrets.token_hex(16), "experiment_id": "daily_energy_ratio_g12_identity_g3_strict_v3", "created_utc": datetime.now(timezone.utc).isoformat()}
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = json.loads(path.read_text(encoding="utf-8"))
            if pid_is_alive(int(current["pid"])):
                raise RuntimeError(f"heavy guard held: {current}")
            path.unlink()
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(owner, stream, sort_keys=True); stream.write("\n")
        return owner
    raise RuntimeError("guard acquisition failed")


def release_guard(path: Path, owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return
    if current.get("pid") == owner.get("pid") and current.get("token") == owner.get("token"):
        path.unlink()


def verify_config() -> dict[str, Any]:
    if sha256(CONFIG) != SIDECAR.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("v2 prereg sidecar mismatch")
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    v2 = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    expected = {
        V2_CONFIG: payload["base_v2"]["sha256"],
        V1_CONFIG: v2["supersedes_and_inherits"]["v1_candidate_contract"]["sha256"],
        INCIDENT: v2["supersedes_and_inherits"]["incident_lock"]["sha256"],
        ROOT / payload["failed_attempt"]["path"]: payload["failed_attempt"]["sha256"],
        REJECTION: "ad00f861f0ff7cfb0d0fafb7928768e1d052abd08e65b09e0623e8662927912f",
        BASE_2023: "9f9d4a45a0dad3c19a80905c021a960f2cf1c813a1709cb575d17c5c78dccb76",
        ROOT / "src/metric.py": "555950e6892a808d9091b4e6749128b9f96888ce0a83270b6959a4e927dc5f1d",
    }
    bad = {str(p): (h, sha256(p)) for p, h in expected.items() if sha256(p) != h}
    if bad:
        raise RuntimeError(f"input identity mismatch: {bad}")
    return payload


def source_lock(out: Path) -> Path:
    sources = [CONFIG, SIDECAR, V2_CONFIG, V2_CONFIG.with_suffix(".sha256"), V1_CONFIG, V1_CONFIG.with_suffix(".sha256"), INCIDENT, INCIDENT.with_suffix(".sha256"), REJECTION, REJECTION.with_suffix(".sha256"), ROOT / "artifacts/postgate/daily_energy_ratio_g12_identity_g3_strict_v2_failed_attempt_1/attempt_manifest.json", ROOT / "scripts/run_daily_energy_ratio_g12.py", ROOT / "tests/test_daily_energy_ratio_g12.py", ROOT / "scripts/run_shared_q07_multiseed.py", ROOT / "src/features.py", ROOT / "src/metric.py"]
    payload = {
        "status": "locked_before_raw_weather_or_label_read_candidate_fit_prediction_score",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [file_record(p) for p in sources],
        "safe_inputs": [file_record(BASE_2023), file_record(RAW_DIR / "info.xlsx")],
        "raw_prefixes_bound_from_preregister_without_suffix_read": json.loads(V2_CONFIG.read_text(encoding="utf-8"))["v2_stage1_weather_source"],
        "full_cache_hashes_not_recomputed": True,
        "permanent_incident": {"prepreregister_2024_weather_cache_read_only": True, "2024_weather_value_cells_materialized": 52704, "2024_label_prediction_metric": 0},
        "label_content_bytes_read": 0,
    }
    path = out / "source_lock_before_inputs.json"; atomic_json(payload, path); return path


def run_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (index - pd.Timedelta(hours=1)).normalize()


def daily_features(baseline: pd.Series, weather: pd.DataFrame, capacity: float) -> pd.DataFrame:
    if not baseline.index.equals(weather.index):
        raise ValueError("baseline/weather indexes differ")
    key = run_key(baseline.index)
    counts = pd.Series(1, index=baseline.index).groupby(key).sum()
    if not counts.eq(24).all():
        raise ValueError("daily run does not contain exactly 24 hours")
    result = pd.DataFrame(index=counts.index)
    result["baseline_daily_mean_cf"] = baseline.groupby(key).mean() / capacity
    series = {
        "ldaps": weather["ldaps__idw__hub_ws"],
        "gfs": weather["gfs__idw__hub_ws"],
        "ldaps_minus_gfs": weather["ldaps__idw__hub_ws"] - weather["gfs__idw__hub_ws"],
    }
    for name, values in series.items():
        grouped = values.groupby(key)
        result[f"{name}_ws_mean" if name in ("ldaps", "gfs") else f"{name}_mean"] = grouped.mean()
        result[f"{name}_ws_std" if name in ("ldaps", "gfs") else f"{name}_std"] = grouped.std(ddof=0)
        for quantile, suffix in ((.25, "q25"), (.5, "q50"), (.75, "q75")):
            column = f"{name}_ws_{suffix}" if name in ("ldaps", "gfs") else f"{name}_{suffix}"
            result[column] = grouped.quantile(quantile)
    angle = 2 * np.pi * (result.index.dayofyear.to_numpy() - 1) / 365.2425
    result["doy_sin1"], result["doy_cos1"] = np.sin(angle), np.cos(angle)
    result["doy_sin2"], result["doy_cos2"] = np.sin(2 * angle), np.cos(2 * angle)
    result = result.loc[:, FEATURE_COLUMNS].astype(np.float64)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("daily features non-finite")
    return result


def daily_target(labels: pd.Series, baseline: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    if not labels.index.equals(baseline.index):
        raise ValueError("label/baseline indexes differ")
    key = run_key(labels.index)
    label_count = labels.groupby(key).count()
    baseline_count = baseline.groupby(key).count()
    denominator = baseline.groupby(key).sum()
    eligible = (label_count == 24) & (baseline_count == 24) & (denominator > 0)
    target = (labels.groupby(key).sum() / denominator).loc[eligible]
    audit = {
        "all_run_count": int(len(eligible)),
        "eligible_run_count": int(eligible.sum()),
        "dropped_run_keys": [str(value) for value in eligible.index[~eligible]],
        "eligible_mask_sha256": hashlib.sha256(eligible.to_numpy(dtype=np.uint8).tobytes()).hexdigest(),
        "eligible_run_key_sha256": hashlib.sha256(target.index.asi8.tobytes()).hexdigest(),
        "target_sha256": hashlib.sha256(target.to_numpy(dtype=np.float64).tobytes()).hexdigest(),
    }
    return target, audit


def make_model() -> Pipeline:
    return Pipeline([("standardize", StandardScaler()), ("ridge", Ridge(alpha=100.0, fit_intercept=True, solver="cholesky", tol=1e-8))])


def factor_from_model(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    raw = model.predict(features)
    factor = 1.0 + 0.25 * (np.clip(raw, 0.90, 1.05) - 1.0)
    return pd.Series(factor, index=features.index, name="effective_factor")


def apply_factor(baseline: pd.Series, factor: pd.Series) -> pd.Series:
    mapped = pd.Series(run_key(baseline.index), index=baseline.index).map(factor)
    if mapped.isna().any():
        raise ValueError("factor mapping incomplete")
    return (baseline * mapped.to_numpy(dtype=np.float64)).rename(baseline.name)


class LabelCursor:
    def __init__(self, path: Path):
        self.path = path; self.stream = path.open("rb", buffering=0); self.rows = 0
        self.digest = hashlib.sha256(); self.header = self.stream.readline(); self.digest.update(self.header)
        if next(csv.reader([self.header.decode("utf-8-sig").strip()])) != ["kst_dtm", *TARGET_COLS]:
            raise ValueError("label header differs")

    def read_initial_q1(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        times: list[pd.Timestamp] = []; values: list[list[float]] = []
        while self.rows < 10920:
            line = self.stream.readline(); self.digest.update(line); self.rows += 1
            ts = pd.Timestamp(line.split(b",", 1)[0].decode("ascii"))
            if ts >= S1["Q1"][0]:
                fields = line.rsplit(b",", 1)[0].decode("utf-8").split(",")
                times.append(ts); values.append([float(value) if value else np.nan for value in fields[1:3]])
        frame = pd.DataFrame(values, index=pd.DatetimeIndex(times, name="forecast_kst_dtm"), columns=GROUPS)
        if not frame.index.equals(S1["Q1"]): raise ValueError("Q1 labels differ")
        return frame, self.record("Q1")

    def read_segment(self, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        index = S1[name]; times=[]; values=[]
        for _ in range(len(index)):
            line=self.stream.readline(); self.digest.update(line); self.rows += 1
            fields=line.rsplit(b",",1)[0].decode("utf-8").split(",")
            times.append(pd.Timestamp(fields[0])); values.append([float(value) if value else np.nan for value in fields[1:3]])
        frame=pd.DataFrame(values,index=pd.DatetimeIndex(times,name="forecast_kst_dtm"),columns=GROUPS)
        if not frame.index.equals(index): raise ValueError(f"{name} labels differ")
        return frame,self.record(name)

    def read_2024(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        times=[]; values=[]
        for _ in range(len(Y2024)):
            line=self.stream.readline(); self.digest.update(line); self.rows += 1
            fields=next(csv.reader([line.decode("utf-8").strip()]))
            times.append(pd.Timestamp(fields[0])); values.append([float(v) if v else np.nan for v in fields[1:]])
        frame=pd.DataFrame(values,index=pd.DatetimeIndex(times,name="forecast_kst_dtm"),columns=TARGET_COLS)
        if not frame.index.equals(Y2024): raise ValueError("2024 labels differ")
        return frame,self.record("2024")

    def record(self, phase: str) -> dict[str, Any]:
        return {"phase": phase, "physical_rows_read": self.rows, "physical_bytes_read": self.stream.tell(), "prefix_sha256": self.digest.hexdigest(), "G3_value_cells_materialized_before_2024": 0}

    def close(self) -> None: self.stream.close()


def metric(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, float]:
    p = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group).as_dict()
    p["total_score"] = 0.5 * (p["one_minus_nmae"] + p["ficr"]); return p


def compare(actual: pd.Series, baseline: pd.Series, candidate: pd.Series, group: str) -> dict[str, Any]:
    b=metric(actual,baseline,group); c=metric(actual,candidate,group)
    return {"baseline":b,"candidate":c,"delta":{k:c[k]-b[k] for k in ("total_score","one_minus_nmae","ficr")}}


def block_fit_predict(block: str, fit_labels: pd.DataFrame, apply_index: pd.DatetimeIndex, features: Mapping[str,pd.DataFrame], baseline: pd.DataFrame, out: Path) -> tuple[pd.DataFrame,pd.DataFrame,Path]:
    candidates=pd.DataFrame(index=apply_index); factors=pd.DataFrame(index=pd.Index(sorted(set(run_key(apply_index))),name="run_date")); models={}; target_audits={}
    for group in GROUPS:
        fit_index=fit_labels.index; y,target_audits[group]=daily_target(fit_labels[group],baseline.loc[fit_index,group])
        model=make_model(); model.fit(features[group].loc[y.index],y.to_numpy())
        factor=factor_from_model(model,features[group].loc[factors.index]); factors[group]=factor
        candidates[group]=apply_factor(baseline.loc[apply_index,group],factor)
        models[group]=model
    model_path=out/f"models/{block}.joblib"; model_path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(models,model_path,compress=3)
    reloaded=joblib.load(model_path)
    for group in GROUPS:
        observed=factor_from_model(reloaded[group],features[group].loc[factors.index])
        if not np.array_equal(observed.to_numpy(),factors[group].to_numpy()): raise AssertionError("reload prediction differs")
    atomic_parquet(candidates,out/f"predictions/{block}_candidate.parquet"); atomic_parquet(factors,out/f"predictions/{block}_factors.parquet")
    lock=out/f"locks/{block}_before_labels.json"; atomic_json({"status":f"{block}_model_candidate_locked_before_{block}_labels","model":file_record(model_path),"candidate":file_record(out/f"predictions/{block}_candidate.parquet"),"factors":file_record(out/f"predictions/{block}_factors.parquet"),"target_masks":target_audits,"next_block_label_cells_read":0},lock)
    return candidates,factors,lock


def main() -> None:
    payload=verify_config()
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    closure=source_lock(OUT); print(f"source_lock={sha256(closure)}",flush=True)
    owner=acquire_guard(HEAVY_GUARD_PATH); atexit.register(release_guard,HEAVY_GUARD_PATH,owner); print(f"guard pid={owner['pid']} token={owner['token']}",flush=True)
    empty=pd.DataFrame(index=pd.date_range("2022-01-01 01:00","2024-01-01 00:00",freq="h",name="forecast_kst_dtm"))
    raw,raw_evidence=shared._read_stage1_raw_features(RAW_DIR,empty)
    base=pd.read_parquet(BASE_2023).loc[Y2023,list(GROUPS)].astype(np.float64)
    features={}
    audit={}
    for group in GROUPS:
        raw_2023=raw[group].loc[Y2023,list(WEATHER_COLUMNS)].astype(np.float64)
        cached=pd.read_parquet(CACHE[group],columns=list(WEATHER_COLUMNS),filters=[("forecast_kst_dtm",">=",Y2023[0].to_pydatetime()),("forecast_kst_dtm","<=",Y2023[-1].to_pydatetime())])
        cached.index=pd.DatetimeIndex(cached.index,name="forecast_kst_dtm")
        exact=cached.index.equals(raw_2023.index) and np.array_equal(cached.to_numpy(),raw_2023.to_numpy())
        if not exact: raise AssertionError(f"{group} postlock raw/cache audit differs")
        audit[group]={"rows":len(cached),"columns":list(cached.columns),"value_bit_exact":True,"logical_filter":"2023 only","parquet_row_groups":1,"prepreregister_2024_weather_cache_read_only":True}
        features[group]=daily_features(base[group],raw_2023,CAPACITY_KWH[group])
    audit_path=OUT/"postlock_2023_raw_cache_audit.json"; atomic_json({"raw":raw_evidence,"audit":audit,"2024_values_in_stage1_feature_frame":0},audit_path)
    cursor=LabelCursor(LABEL_PATH)
    q1,q1rec=cursor.read_initial_q1(); q1lock=OUT/"locks/Q1_fit_labels.json"; atomic_json({"access":q1rec,"future_label_cells":0},q1lock)
    predictions=[]; factors=[]; locks=[]; labels={"Q1":q1}
    for block,fit_names in (("Q2",("Q1",)),("Q3",("Q1","Q2")),("Q4",("Q1","Q2","Q3"))):
        fit=pd.concat([labels[n] for n in fit_names])
        pred,fac,lock=block_fit_predict(block,fit,S1[block],features,base,OUT); predictions.append(pred); factors.append(fac); locks.append(lock)
        segment,record=cursor.read_segment(block); labels[block]=segment
        atomic_json({"candidate_lock":file_record(lock),"access_after_lock":record},OUT/f"locks/{block}_label_access.json")
    stitched=pd.concat(predictions).sort_index(); stitched_path=OUT/"predictions/stage1_stitched_candidate.parquet"; atomic_parquet(stitched,stitched_path)
    label_all=pd.concat([labels[n] for n in ("Q1","Q2","Q3","Q4")]).sort_index()
    comparisons={}; checks=[]
    for group in GROUPS:
        comparisons[group]={}
        for name,index in S1_SLICES.items():
            c=compare(label_all.loc[index,group],base.loc[index,group],stitched.loc[index,group],group); comparisons[group][name]=c; checks.append(c["delta"]["total_score"]>0)
        full=comparisons[group]["Q2_Q4_full"]["delta"]; checks.extend([full["one_minus_nmae"]>=0,full["ficr"]>=0])
    passed=bool(all(checks)); result={"status":"stage1_promoted" if passed else "rejected_on_stage1","passed":passed,"comparisons":comparisons,"prepreregister_2024_weather_cache_read_only":True,"2024_label_prediction_metric":0,"2025_access":0}
    s1path=OUT/"stage1_results.json"; atomic_json(result,s1path); atomic_json({"status":result["status"],"stage1_results":file_record(s1path),"stage2_access_authorized":passed},OUT/("stage1_promotion_lock.json" if passed else "rejection.json"))
    if passed:
        baseline24={name:pd.read_parquet(path).loc[Y2024,list(TARGET_COLS)].astype(np.float64) for name,path in BASE_2024.items()}
        weather24={g:pd.read_parquet(CACHE[g],columns=list(WEATHER_COLUMNS),filters=[("forecast_kst_dtm",">=",Y2024[0].to_pydatetime()),("forecast_kst_dtm","<=",Y2024[-1].to_pydatetime())]) for g in GROUPS}
        factors24=pd.DataFrame(index=pd.Index(sorted(set(run_key(Y2024))),name="run_date")); models={}
        for group in GROUPS:
            weather24[group].index=pd.DatetimeIndex(weather24[group].index,name="forecast_kst_dtm")
            f23=features[group]; y,_=daily_target(label_all[group],base[group]); model=make_model(); model.fit(f23.loc[y.index],y.to_numpy()); models[group]=model
            f24=daily_features(baseline24["primary_corrected_v3"][group],weather24[group],CAPACITY_KWH[group]); factors24[group]=factor_from_model(model,f24)
        mpath=OUT/"models/stage2_refit.joblib"; joblib.dump(models,mpath,compress=3); fpath=OUT/"predictions/stage2_factors.parquet"; atomic_parquet(factors24,fpath)
        candidates24={}
        for name,bline in baseline24.items():
            cand=bline.copy()
            for group in GROUPS: cand[group]=apply_factor(bline[group],factors24[group])
            if not np.array_equal(cand["kpx_group_3"].to_numpy(),bline["kpx_group_3"].to_numpy()): raise AssertionError("G3 identity differs")
            candidates24[name]=cand; atomic_parquet(cand,OUT/f"predictions/stage2_{name}.parquet")
        s2lock=OUT/"stage2_candidate_lock_before_2024_labels.json"; atomic_json({"model":file_record(mpath),"factors":file_record(fpath),"candidates":{n:file_record(OUT/f"predictions/stage2_{n}.parquet") for n in candidates24},"2024_label_cells_read":0},s2lock)
        labels24,record24=cursor.read_2024(); atomic_json({"candidate_lock":file_record(s2lock),"label_access":record24},OUT/"stage2_label_access.json")
        comps={}; passed2=True
        for bname,bline in baseline24.items():
            comps[bname]={"groups":{},"mixed":{}}
            for group in GROUPS:
                comps[bname]["groups"][group]={}
                for sname,index in S2_SLICES.items():
                    c=compare(labels24.loc[index,group],bline.loc[index,group],candidates24[bname].loc[index,group],group); comps[bname]["groups"][group][sname]=c; passed2 &= c["delta"]["total_score"]>0
                full=comps[bname]["groups"][group]["full"]["delta"]; passed2 &= full["one_minus_nmae"]>=0 and full["ficr"]>=0
            for sname,index in S2_SLICES.items():
                bd=score_details(labels24.loc[index],bline.loc[index]).as_dict(); cd=score_details(labels24.loc[index],candidates24[bname].loc[index]).as_dict(); delta={k:cd[k]-bd[k] for k in ("total_score","one_minus_nmae","ficr")}; comps[bname]["mixed"][sname]={"baseline":bd,"candidate":cd,"delta":delta}; passed2 &= delta["total_score"]>0
            full=comps[bname]["mixed"]["full"]["delta"]; passed2 &= full["one_minus_nmae"]>=0 and full["ficr"]>=0
        s2={"status":"promoted_for_separate_final_lock_only" if passed2 else "rejected_on_stage2_dual_gate","passed":bool(passed2),"comparisons":comps,"G3_identity":True,"2025_access":0}; atomic_json(s2,OUT/"stage2_results.json"); atomic_json({"status":s2["status"],"stage2_results":file_record(OUT/"stage2_results.json")},OUT/("promotion_lock.json" if passed2 else "stage2_rejection.json"))
    cursor.close()
    files=[p for p in OUT.rglob("*") if p.is_file()]
    manifest=OUT/"manifest.json"; atomic_json({"experiment_id":payload["experiment_id"],"preregister":file_record(CONFIG),"outputs_excluding_manifest":[file_record(p) for p in files],"stage1_passed":passed,"prepreregister_2024_weather_cache_read_only":True,"2024_label_prediction_metric_before_stage1_pass":0,"2025_access":0,"public_scale_artifacts":0,"csv_created":False},manifest)
    (OUT/"manifest.sha256").write_text(f"{sha256(manifest)}  manifest.json\n",encoding="ascii")
    print(f"stage1_passed={passed} results={sha256(s1path)} manifest={sha256(manifest)}",flush=True)
    release_guard(HEAVY_GUARD_PATH,owner)


if __name__ == "__main__": main()
