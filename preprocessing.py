"""
preprocessing.py — Módulo 1 do projeto CGMacros (Integrante 1)
================================================================

Transforma os arquivos brutos do CGMacros em tabelas organizadas que os outros
módulos (rede neural e algoritmo genético) consomem.

Fluxo (cada etapa é uma função que pode ser usada isoladamente nos notebooks):

    arquivos brutos  ──►  load_participant_csv  ──►  clean_participant  (série por minuto, dias D1..Dn)
                                                         │
                          ┌──────────────────────────────┼───────────────────────┐
                          ▼                              ▼                       ▼
                  build_daily_features             extract_meals            load_bio / clean_bio
                  (1 linha = participante×dia)    (1 linha = refeição)      (1 linha = participante)
                          │                                                      │
                          └────────────► build_wide_dataset  ◄───────────────────┘
                                         (1 linha = participante: D1-D6 → D7-D9 → alvo D9)
                                         build_day_level_dataset (1 linha = participante×dia)
                                                   │
                                   split_participants + fit_preprocessor
                                   (treino/validação/teste POR PARTICIPANTE, escala só com o treino)

Uso pela linha de comando:
    python preprocessing.py --raw data/raw --out data/processed

Uso por outros módulos:
    from preprocessing import load_processed, load_preprocessor
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Constantes do projeto
# --------------------------------------------------------------------------------------
GLUCOSE_MIN, GLUCOSE_MAX = 40.0, 400.0     # limites do sensor (dicionário de dados)
TIR_LOW, TIR_HIGH = 70.0, 180.0            # faixa-alvo clínica de glicose (mg/dL)
ACTIVE_METS = 3.0                          # >= 3 METs = atividade moderada
INPUT_DAYS = (1, 2, 3, 4, 5, 6)            # D1..D6: dados de entrada
EVAL_DAYS = (7, 8, 9)                      # D7..D9: dias a serem recomendados / avaliados
TARGET_DAY = 9                             # dia cuja glicose queremos reduzir
MEAL_TYPES = ("breakfast", "lunch", "dinner")   # refeições principais (1 por dia)
SNACK = "snack"                                  # lanches (0..n por dia; 'snack', 'snacks', 'snack 1' -> 'snack')
NUTRIENTS = ("calories", "carbs", "protein", "fat", "fiber")
# Tetos de plausibilidade POR REFEIÇÃO. O dataset real tem erros de digitação (ex.: 2830 g de fibra,
# 761 g de carboidrato). Valores acima do teto viram "faltante" e são imputados (ver clean_participant).
MEAL_VALUE_CAPS = {"calories": 2500, "carbs": 300, "protein": 200, "fat": 200, "fiber": 100}

# nome original (minúsculo, sem espaços extras) -> nome padronizado
_COLUMN_ALIASES = {
    "timestamp": "timestamp", "libre gl": "libre_gl", "dexcom gl": "dexcom_gl", "hr": "hr",
    "calories (activity)": "cal_activity", "mets": "mets", "meal type": "meal_type",
    "calories": "calories", "carbs": "carbs", "protein": "protein", "fat": "fat", "fiber": "fiber",
    "amount consumed": "amount_consumed", "image path": "image_path",
}
_NUMERIC_COLS = ["libre_gl", "dexcom_gl", "hr", "cal_activity", "mets", *NUTRIENTS, "amount_consumed"]

# Features de COMPORTAMENTO do dia (o que o algoritmo genético poderá controlar)
BEHAVIOR_FEATURES = (
    ["cal_total", "carbs_total", "protein_total", "fat_total", "fiber_total", "n_meals", "n_snacks"]
    + [f"{m}_{n}" for m in (*MEAL_TYPES, SNACK) for n in NUTRIENTS]
    + [f"{m}_hour" for m in MEAL_TYPES]
    + ["hr_mean", "mets_mean", "active_min", "cal_activity_sum"]
)
# Features de RESPOSTA glicêmica do dia
GLUCOSE_FEATURES = ["gl_mean", "gl_std", "gl_min", "gl_max", "gl_cv", "tir", "tar", "tbr"]


# --------------------------------------------------------------------------------------
# 1. Leitura
# --------------------------------------------------------------------------------------
def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def normalize_meal_type(s: pd.Series) -> pd.Series:
    """minúsculas, sem espaços; qualquer variante de lanche ('snack', 'snacks', 'snack 1') -> 'snack'."""
    mt = s.astype("string").str.strip().str.lower()
    mt = mt.where(~mt.isin(["", "nan", "none"]), pd.NA)
    return mt.where(~mt.str.startswith("snack", na=False), SNACK)


def find_participant_files(raw_dir: str | Path) -> dict[int, Path]:
    """Procura recursivamente arquivos 'CGMacros-0XX.csv' (ignora o dicionário de dados)."""
    files = {}
    for p in sorted(Path(raw_dir).rglob("CGMacros-*.csv")):
        m = re.fullmatch(r"CGMacros-(\d+)\.csv", p.name)
        if m:
            files[int(m.group(1))] = p
    return files


def find_bio_file(raw_dir: str | Path) -> Path | None:
    for p in Path(raw_dir).rglob("*.csv"):
        if p.name.lower() == "bio.csv":
            return p
    return None


def _parse_timestamps(s: pd.Series) -> pd.Series:
    """Formato documentado: Mês/Dia/Ano HH:MM; cai para um parser flexível (ISO etc.) se necessário."""
    s = s.astype("string").str.strip()
    out = pd.to_datetime(s, format="%m/%d/%Y %H:%M", errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        out[bad] = pd.to_datetime(s[bad], format="mixed", errors="coerce")
    return out


def load_participant_csv(path: str | Path) -> pd.DataFrame:
    """Lê o CSV de um participante e padroniza os nomes das colunas."""
    df = pd.read_csv(path)
    df.columns = [_COLUMN_ALIASES.get(_norm(c), _norm(c).replace(" ", "_")) for c in df.columns]
    df = df.loc[:, [not str(c).startswith("unnamed") for c in df.columns]]   # descarta coluna de índice sem nome
    if "timestamp" not in df.columns:
        raise ValueError(f"{path}: coluna 'Timestamp' não encontrada. Colunas: {list(df.columns)}")
    for c in _NUMERIC_COLS + ["meal_type"]:
        if c not in df.columns:
            df[c] = np.nan  # coluna ausente neste arquivo
    df["timestamp"] = _parse_timestamps(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    return df.drop(columns=[c for c in ("image_path",) if c in df.columns])


# --------------------------------------------------------------------------------------
# 2. Limpeza (série por minuto)
# --------------------------------------------------------------------------------------
def _interp_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """Interpola no tempo apenas buracos de até `max_gap` minutos; buracos maiores continuam NaN."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    filled = s.interpolate(method="time", limit_area="inside")
    return filled.where(~(isna & (run_len > max_gap)), np.nan)


def clean_participant(raw: pd.DataFrame, participant_id: int, glucose_source: str = "consensus",
                      max_gap_glucose: int = 30, max_gap_other: int = 10,
                      day_mode: str = "first_full", scale_by_consumed: bool = False,
                      min_coverage: float = 0.8):
    """Limpa um participante e devolve (DataFrame por minuto, dict de estatísticas de qualidade).

    Passos:
      1. arredonda timestamps ao minuto e junta linhas duplicadas;
      2. cria uma grade regular de 1 minuto (assim 'buraco de 30 min' é medido corretamente);
      3. glicose fora de [40, 400] vira NaN; interpola só buracos curtos (<= max_gap_glucose min);
      4. unifica Libre e Dexcom em `glucose` (glucose_source):
           'consensus' (padrão): média dos dois sensores, alinhados por participante. No dataset real o Libre lê
                       ~20-30 mg/dL ABAIXO e o Dexcom ~20-40 ACIMA da glicemia de ponta de dedo (bio.csv); a média
                       dos dois fica a poucos mg/dL dela. Nos minutos em que só um sensor existe, ele é deslocado
                       em metade do desvio mediano (Dexcom - Libre) do participante, para a série não dar
                       "degraus" quando um sensor some. Ver `fingerstick_validation`.
           'libre_filled' | 'mean' | 'libre' | 'dexcom': alternativas (um sensor só, média simples...);
      5. atividade: HR e calorias de atividade interpolados em buracos curtos. METs: alguns arquivos têm `METs`
         (valor x 10) e outros `Intensity` (0/1) no lugar; nesses, METs é estimado de `Calories (Activity)`
         (kcal/min = METs x kcal-por-MET do participante; 1 MET = 1º percentil do gasto por minuto);
      6. refeições: tipos normalizados (lanche); macros acima de MEAL_VALUE_CAPS (erros de digitação) viram
         faltantes e são imputados pela mediana do mesmo tipo de refeição do participante; nas linhas SEM
         refeição o valor é 0 (não é dado faltante, é "não comeu");
      7. numera os dias D1..Dn por participante (day_mode):
           'first_full' (padrão): D1 = primeiro dia de calendário com glicose em >= min_coverage do dia.
                       O registro começa no meio do dia, então o 1º dia de calendário é parcial; descartá-lo
                       deixa D1..D9 completos. Minutos anteriores a D1 recebem day = 0 (ficam só no minute_clean).
           'calendar': D1 = primeira data (pode ser parcial).
           'from_start': blocos de 24 h desde o início (quebra as refeições ao meio; não recomendado).
    """
    df = raw.copy()
    stats = {"participant_id": participant_id, "raw_rows": len(df)}
    df["timestamp"] = df["timestamp"].dt.floor("min")
    for c in _NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["meal_type"] = normalize_meal_type(df["meal_type"])
    df["amount_consumed"] = df["amount_consumed"].clip(lower=0, upper=100)  # dicionário: 0-100 %

    stats["duplicate_timestamps"] = int(df.duplicated("timestamp").sum())
    df = df.sort_values("timestamp", kind="stable").groupby("timestamp").first()  # 1º valor não-nulo de cada coluna

    full = pd.date_range(df.index.min(), df.index.max(), freq="min")
    stats["minutes_in_grid"] = len(full)
    stats["minutes_missing_rows"] = len(full) - len(df)
    df = df.reindex(full)
    df.index.name = "timestamp"

    # --- glicose
    for c in ("libre_gl", "dexcom_gl"):
        raw_s = df[c]
        stats[f"{c}_pct_missing_raw"] = round(100 * raw_s.isna().mean(), 2)
        out_of_range = raw_s.notna() & ((raw_s < GLUCOSE_MIN) | (raw_s > GLUCOSE_MAX))
        stats[f"{c}_out_of_range"] = int(out_of_range.sum())
        df[c] = _interp_short_gaps(raw_s.where(~out_of_range), max_gap_glucose)
    offset = (df["dexcom_gl"] - df["libre_gl"]).median()   # NaN se os sensores nunca coincidem
    stats["dexcom_minus_libre_median"] = round(float(offset), 2) if np.isfinite(offset) else np.nan
    if glucose_source == "consensus":
        if np.isfinite(offset):
            both = pd.concat([df["libre_gl"] + offset / 2, df["dexcom_gl"] - offset / 2], axis=1)
            df["glucose"] = both.mean(axis=1, skipna=True)   # com os 2 sensores = (Libre + Dexcom) / 2
        else:                                                # sem sobreposição: usa o que existir
            df["glucose"] = df["libre_gl"].fillna(df["dexcom_gl"])
    elif glucose_source == "libre_filled":
        df["glucose"] = df["libre_gl"].fillna(df["dexcom_gl"] - offset)
    elif glucose_source == "mean":
        df["glucose"] = df[["libre_gl", "dexcom_gl"]].mean(axis=1, skipna=True)
    elif glucose_source in ("libre", "dexcom"):
        df["glucose"] = df[f"{glucose_source}_gl"]
    else:
        raise ValueError("glucose_source deve ser 'consensus', 'libre_filled', 'mean', 'libre' ou 'dexcom'")
    # o alinhamento (±offset/2) pode empurrar leituras próximas dos extremos para fora da faixa dos sensores
    df["glucose"] = df["glucose"].clip(GLUCOSE_MIN, GLUCOSE_MAX)
    stats["glucose_pct_missing"] = round(100 * df["glucose"].isna().mean(), 2)

    # --- atividade (harmoniza arquivos com METs e arquivos com Intensity)
    cal = df["cal_activity"]
    mets = df["mets"] / 10.0                                 # o arquivo guarda METs x 10
    has_mets = bool(mets.notna().mean() > 0.2)
    if has_mets:                                             # kcal por MET medido neste participante
        kcal_per_met = (cal / mets).where(mets > 0).replace([np.inf, -np.inf], np.nan).median()
        stats["mets_source"] = "METs"
    else:                                                    # 1 MET = gasto mínimo por minuto (repouso)
        pos = cal[cal > 0]
        kcal_per_met = pos.quantile(0.01) if len(pos) else np.nan
        stats["mets_source"] = "estimated_from_calories" if np.isfinite(kcal_per_met) else "missing"
    stats["kcal_per_met"] = round(float(kcal_per_met), 4) if np.isfinite(kcal_per_met) else np.nan
    # 0 kcal/min não existe com a pessoa viva (repouso ~1 kcal/min): é ausência de dado, não 0 METs
    mets_est = (cal.where(cal > 0) / kcal_per_met if np.isfinite(kcal_per_met) and kcal_per_met > 0
                else pd.Series(np.nan, index=df.index))
    df["mets"] = mets.fillna(mets_est) if has_mets else mets_est
    for c in ("hr", "mets", "cal_activity"):
        df[c] = _interp_short_gaps(df[c], max_gap_other)
    stats["hr_pct_missing"] = round(100 * df["hr"].isna().mean(), 2)
    stats["mets_pct_missing"] = round(100 * df["mets"].isna().mean(), 2)

    # --- refeições
    is_meal = df["meal_type"].notna()
    stats["meal_values_capped"] = 0
    for c, cap in MEAL_VALUE_CAPS.items():
        bad = df[c] > cap
        stats[f"capped_{c}"] = int(bad.sum())
        stats["meal_values_capped"] += int(bad.sum())
        df.loc[bad, c] = np.nan
    stats["meal_values_imputed"] = 0
    for c in NUTRIENTS:  # refeição com macro faltante/erro -> mediana do mesmo tipo de refeição do participante
        miss = is_meal & df[c].isna()
        if miss.any():
            by_type = df.loc[is_meal].groupby("meal_type")[c].median()
            fill = df.loc[miss, "meal_type"].map(by_type).astype(float)
            fill = fill.fillna(df.loc[is_meal, c].median())
            df.loc[miss, c] = fill
            stats["meal_values_imputed"] += int(miss.sum())
    if scale_by_consumed:
        factor = df["amount_consumed"].fillna(100.0) / 100.0
        for c in NUTRIENTS:
            df[c] = df[c] * factor
    for c in NUTRIENTS:
        df[c] = df[c].fillna(0.0)
    stats["n_meals"] = int(is_meal.sum())

    # --- dias
    df["date"] = df.index.normalize()
    if day_mode == "calendar":
        df["day"] = (df["date"] - df["date"].min()).dt.days + 1
    elif day_mode == "first_full":
        cov = df["glucose"].notna().groupby(df["date"]).sum() / 1440
        full_dates = cov.index[cov >= min_coverage]
        d0 = full_dates.min() if len(full_dates) else df["date"].min()
        df["day"] = ((df["date"] - d0).dt.days + 1).clip(lower=0)
    elif day_mode == "from_start":
        df["day"] = ((df.index - df.index.min()) // pd.Timedelta(days=1)) + 1
    else:
        raise ValueError("day_mode deve ser 'first_full', 'calendar' ou 'from_start'")
    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    df.insert(0, "participant_id", participant_id)
    stats["n_days"] = int(df["day"].max())
    stats["minutes_before_D1"] = int((df["day"] == 0).sum())
    stats["start"], stats["end"] = str(df.index.min()), str(df.index.max())
    return df.reset_index(), stats


# --------------------------------------------------------------------------------------
# 3. Features por dia
# --------------------------------------------------------------------------------------
def build_daily_features(minute: pd.DataFrame, min_coverage: float = 0.8,
                         min_activity_coverage: float = 0.5) -> pd.DataFrame:
    """Uma linha por (participante, dia) com comportamento e resposta glicêmica.

    `glucose_coverage` = fração do dia (1440 min) com glicose válida; `day_complete` = cobertura >= min_coverage
    (com day_mode='first_full', D1..D9 costumam ser completos; só o último dia é parcial).

    Atividade (smartwatch): em alguns participantes o relógio some por horas/dias (até ~50 % de HR faltando).
    Se a cobertura do dia (`activity_coverage` para HR, `mets_coverage` para METs/calorias) é < min_activity_coverage,
    as features correspondentes ficam NaN (serão imputadas com a mediana do treino); caso contrário, `active_min` e
    `cal_activity_sum` são normalizados para o dia inteiro (média dos minutos observados × minutos do dia).
    """
    rows = []
    for (pid, day), g in minute[minute["day"] >= 1].groupby(["participant_id", "day"], sort=True):  # day 0 = antes de D1
        r = {"participant_id": pid, "day": int(day), "date": g["date"].iloc[0]}
        gl = g["glucose"].dropna()
        r["n_minutes"] = len(g)
        r["glucose_coverage"] = len(gl) / 1440
        r["day_complete"] = r["glucose_coverage"] >= min_coverage
        if len(gl):
            r.update(gl_mean=gl.mean(), gl_std=gl.std(), gl_min=gl.min(), gl_max=gl.max(),
                     gl_cv=100 * gl.std() / gl.mean(),
                     tir=100 * gl.between(TIR_LOW, TIR_HIGH).mean(),
                     tar=100 * (gl > TIR_HIGH).mean(), tbr=100 * (gl < TIR_LOW).mean())
        else:
            r.update({k: np.nan for k in GLUCOSE_FEATURES})

        meals = g[g["meal_type"].notna()]
        r["n_meals"] = len(meals)
        r["n_snacks"] = int((meals["meal_type"] == SNACK).sum())
        for n in NUTRIENTS:
            r[{"calories": "cal_total"}.get(n, f"{n}_total")] = meals[n].sum()
        for m in (*MEAL_TYPES, SNACK):
            sub = meals[meals["meal_type"] == m]
            for n in NUTRIENTS:
                r[f"{m}_{n}"] = sub[n].sum()
            if m != SNACK:
                r[f"{m}_hour"] = sub["minute_of_day"].iloc[0] / 60 if len(sub) else np.nan
        r["first_meal_hour"] = meals["minute_of_day"].min() / 60 if len(meals) else np.nan
        r["last_meal_hour"] = meals["minute_of_day"].max() / 60 if len(meals) else np.nan

        r["activity_coverage"] = g["hr"].notna().mean()
        r["mets_coverage"] = g["mets"].notna().mean()
        if r["activity_coverage"] >= min_activity_coverage:
            r["hr_mean"], r["hr_max"] = g["hr"].mean(), g["hr"].max()
        else:
            r.update(hr_mean=np.nan, hr_max=np.nan)
        if r["mets_coverage"] >= min_activity_coverage:
            mets = g["mets"].dropna()
            r["mets_mean"] = mets.mean()
            r["active_min"] = (mets >= ACTIVE_METS).mean() * len(g)
            r["cal_activity_sum"] = g["cal_activity"].mean() * len(g)
        else:
            r.update(mets_mean=np.nan, active_min=np.nan, cal_activity_sum=np.nan)
        rows.append(r)

    daily = pd.DataFrame(rows)
    daily["split"] = np.where(daily["day"].isin(INPUT_DAYS), "input",
                              np.where(daily["day"].isin(EVAL_DAYS), "eval", "extra"))
    return daily


def split_input_eval(daily: pd.DataFrame):
    """Separa D1–D6 (entrada) de D7–D9 (avaliação)."""
    return (daily[daily["day"].isin(INPUT_DAYS)].copy(), daily[daily["day"].isin(EVAL_DAYS)].copy())


# --------------------------------------------------------------------------------------
# 4. Refeições e resposta pós-prandial
# --------------------------------------------------------------------------------------
def extract_meals(minute: pd.DataFrame, pre_window: int = 15, post_window: int = 120,
                  peak_window: int = 180) -> pd.DataFrame:
    """Uma linha por refeição, com macros e resposta glicêmica (baseline, pico, iAUC de 2 h)."""
    out = []
    for pid, g in minute.groupby("participant_id", sort=True):
        g = g.reset_index(drop=True)
        gl = g["glucose"].to_numpy(float)
        idx = np.flatnonzero(g["meal_type"].notna().to_numpy())
        for k, i in enumerate(idx):
            pre = gl[max(0, i - pre_window): i + 1]
            base = pre[np.isfinite(pre)].mean() if np.isfinite(pre).any() else np.nan
            post = gl[i: i + peak_window + 1]
            fin = np.isfinite(post)
            peak = post[fin].max() if fin.any() else np.nan
            t_peak = int(np.nanargmax(post)) if fin.any() else np.nan
            w = gl[i: i + post_window + 1]
            ok = np.isfinite(w).mean() >= 0.8 and np.isfinite(base)
            iauc = float(np.trapezoid(np.clip(np.nan_to_num(w - base, nan=0.0), 0, None))) if ok else np.nan
            gap = int(idx[k + 1] - i) if k + 1 < len(idx) else np.nan
            out.append({
                "participant_id": pid, "day": int(g.at[i, "day"]), "timestamp": g.at[i, "timestamp"],
                "meal_type": g.at[i, "meal_type"], "hour": g.at[i, "minute_of_day"] / 60,
                **{n: g.at[i, n] for n in NUTRIENTS},
                "glucose_baseline": base, "glucose_peak": peak,
                "delta_peak": peak - base if np.isfinite(base) else np.nan,
                "minutes_to_peak": t_peak, "iauc_2h": iauc,
                "mean_2h": float(np.nanmean(w)) if np.isfinite(w).any() else np.nan,
                "minutes_to_next_meal": gap, "overlaps_next_meal": bool(gap < post_window) if gap == gap else False,
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# 5. Dados do participante (bio)
# --------------------------------------------------------------------------------------
_BIO_ALIASES = {
    "age": "age", "gender": "gender", "bmi": "bmi", "body weight": "weight_lb", "height": "height_in",
    "self-identify": "ethnicity", "a1c pdl (lab)": "a1c", "fasting glu - pdl (lab)": "fasting_glu",
    "insulin": "insulin", "triglycerides": "triglycerides", "cholesterol": "cholesterol", "hdl": "hdl",
    "non hdl": "non_hdl", "ldl (cal)": "ldl", "vldl (cal)": "vldl", "cho/hdl ratio": "chol_hdl_ratio",
    "#1 contour fingerstick glu": "fingerstick_1", "#2 contour fingerstick glu": "fingerstick_2",
    "#3 contour fingerstick glu": "fingerstick_3",
}
_ID_NAMES = ("subject", "participant", "participant_id", "participant id", "id", "subject id")


def load_bio(path: str | Path) -> pd.DataFrame:
    """Lê e limpa o bio.csv (1 linha por participante)."""
    bio = pd.read_csv(path)
    bio.columns = [_norm(c) for c in bio.columns]
    cols = list(bio.columns)
    for k in (1, 2, 3):                       # cada glicemia de ponta de dedo é seguida pela coluna do horário
        name = f"#{k} contour fingerstick glu"
        if name in cols and cols.index(name) + 1 < len(cols) and cols[cols.index(name) + 1].startswith("time"):
            bio[f"fingerstick_{k}_time"] = bio[cols[cols.index(name) + 1]].astype("string").str.strip()
    id_col = next((c for c in bio.columns if c in _ID_NAMES), None)
    if id_col is None:
        raise ValueError(f"bio.csv: não achei a coluna do participante. Colunas: {list(bio.columns)}. "
                         "Renomeie-a para 'subject' ou ajuste _ID_NAMES em preprocessing.py.")
    bio["participant_id"] = bio[id_col].astype("string").str.extract(r"(\d+)")[0].astype(int)
    bio = bio.rename(columns=_BIO_ALIASES)
    keep = (["participant_id"] + [v for v in _BIO_ALIASES.values() if v in bio.columns]
            + [f"fingerstick_{k}_time" for k in (1, 2, 3) if f"fingerstick_{k}_time" in bio.columns])
    bio = bio[keep].copy()  # descarta colunas de horário ("Time (t)", "Collection time")

    for c in bio.columns.drop(["participant_id", "gender", "ethnicity", "fingerstick_1_time", "fingerstick_2_time",
                               "fingerstick_3_time"], errors="ignore"):
        bio[c] = pd.to_numeric(bio[c], errors="coerce")
    # valores-erro documentados no dicionário
    for col, bad in (("ldl", 800), ("vldl", 400), ("chol_hdl_ratio", 400)):
        if col in bio:
            bio.loc[bio[col] == bad, col] = np.nan
    # A1c: o dicionário diz mmol/mol, mas 4.6-8.5 é %; converte se vier em mmol/mol (IFCC -> NGSP)
    if "a1c" in bio:
        bio["a1c"] = np.where(bio["a1c"] > 20, bio["a1c"] / 10.929 + 2.15, bio["a1c"])
        bio["group_a1c"] = pd.cut(bio["a1c"], [-np.inf, 5.7, 6.5, np.inf], right=False,
                                  labels=["healthy", "prediabetes", "t2d"]).astype("string")
    if {"insulin", "fasting_glu"} <= set(bio.columns):
        bio["homa_ir"] = bio["insulin"] * bio["fasting_glu"] / 405
    if "gender" in bio:
        bio["gender_male"] = (bio["gender"].astype("string").str.strip().str.upper() == "M").astype(int)
    if "ethnicity" in bio:
        e = bio["ethnicity"].astype("string").str.lower()
        eth = np.select([e.str.contains("african", na=False), e.str.contains("hispanic", na=False),
                         e.str.contains("white", na=False)], ["black", "hispanic", "white"], default="other")
        for k in ("black", "hispanic", "white", "other"):
            bio[f"eth_{k}"] = (eth == k).astype(int)
    return bio.drop_duplicates("participant_id").reset_index(drop=True)


BIO_FEATURES = ["age", "gender_male", "bmi", "weight_lb", "height_in", "a1c", "fasting_glu", "insulin",
                "triglycerides", "cholesterol", "hdl", "non_hdl", "ldl", "vldl", "chol_hdl_ratio", "homa_ir",
                "eth_black", "eth_hispanic", "eth_white", "eth_other"]


def _hhmm_to_timedelta(x):
    x = str(x).strip()
    if x.count(":") == 1:
        x += ":00"
    return pd.to_timedelta(x, errors="coerce")


def fingerstick_validation(minute: pd.DataFrame, bio: pd.DataFrame, window: int = 2):
    """Compara os sensores com as 3 glicemias de ponta de dedo do `bio.csv` (manhã do 1º dia de registro).

    Para cada medição, usa a média dos sensores em ±`window` minutos e só considera medições em que Libre E
    Dexcom existem (o Dexcom só começa ~1 h depois do Libre). Devolve (detalhe por medição, resumo por série).
    Atenção: são poucas medições (<= 3 por participante), no início do uso do sensor; serve para comparar a
    ORDEM DE GRANDEZA do viés de cada sensor, não para calibrar com precisão.
    """
    rows = []
    for pid, g in minute.groupby("participant_id"):
        b = bio[bio["participant_id"] == pid]
        if b.empty:
            continue
        b, g = b.iloc[0], g.set_index("timestamp").sort_index()
        day0 = g.index.min().normalize()
        for k in (1, 2, 3):
            val, tstr = b.get(f"fingerstick_{k}"), b.get(f"fingerstick_{k}_time")
            if val is None or tstr is None or pd.isna(val) or pd.isna(tstr):
                continue
            td = _hhmm_to_timedelta(tstr)
            if pd.isna(td):
                continue
            t = day0 + td
            w = g.loc[t - pd.Timedelta(minutes=window): t + pd.Timedelta(minutes=window)]
            if w.empty:
                continue
            rows.append({"participant_id": pid, "k": k, "time": t, "fingerstick": float(val),
                         "libre": w["libre_gl"].mean(), "dexcom": w["dexcom_gl"].mean(),
                         "pipeline_glucose": w["glucose"].mean()})
    det = pd.DataFrame(rows)
    if det.empty:
        return det, pd.DataFrame()
    det = det.dropna(subset=["libre", "dexcom"]).copy()
    det["midpoint"] = (det["libre"] + det["dexcom"]) / 2
    out = []
    for name in ("libre", "dexcom", "midpoint", "pipeline_glucose"):
        err = det[name] - det["fingerstick"]
        out.append({"series": name, "n": len(det), "bias_mg_dl": err.mean(), "mae_mg_dl": err.abs().mean(),
                    "rmse_mg_dl": float(np.sqrt((err ** 2).mean())), "corr": det[name].corr(det["fingerstick"])})
    return det, pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# 6. Tabelas para modelagem
# --------------------------------------------------------------------------------------
def build_participants_table(daily: pd.DataFrame, stats: pd.DataFrame, bio: pd.DataFrame | None,
                             min_coverage: float = 0.8) -> pd.DataFrame:
    """Resumo por participante + elegibilidade (precisa ter D1..D9 e D9 com glicose suficiente)."""
    rows = []
    need = set(range(1, TARGET_DAY + 1))
    for pid, g in daily.groupby("participant_id"):
        days = set(g["day"])
        d9 = g[g["day"] == TARGET_DAY]
        has_all = need <= days
        d9_ok = bool(len(d9) and d9["glucose_coverage"].iloc[0] >= min_coverage)
        reason = "" if has_all and d9_ok else ("faltam dias entre D1 e D9" if not has_all else "D9 com pouca glicose")
        rows.append({"participant_id": pid, "n_days": len(days), "has_D1_D9": has_all, "d9_coverage_ok": d9_ok,
                     "eligible": has_all and d9_ok, "reason_ineligible": reason,
                     "mean_coverage_D1_D9": g[g["day"] <= 9]["glucose_coverage"].mean()})
    p = pd.DataFrame(rows)
    p = p.merge(stats[["participant_id", "start", "end", "n_meals"]], on="participant_id", how="left")
    if bio is not None:
        p = p.merge(bio, on="participant_id", how="left")
    return p


def build_wide_dataset(daily: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    """Formulação exata do projeto: 1 linha por participante elegível.

        entradas:  resumo de D1–D6 (médias dos dias completos) + glicose média de cada dia + dados do participante
                   + comportamento planejado para D7, D8 e D9  (o que o AG vai variar)
        alvo:      target_d9_gl_mean (glicose média no D9), além de target_d9_tir e target_d9_gl_max
    """
    elig = participants.loc[participants["eligible"], "participant_id"]
    d = daily[daily["participant_id"].isin(elig)].set_index(["participant_id", "day"])
    rows = []
    for pid in elig:
        row = {"participant_id": pid}
        hist = d.loc[pid].loc[list(INPUT_DAYS)]
        # D1 costuma ser parcial (registro não começa à meia-noite): resumos usam só dias completos
        ref = hist[hist["day_complete"]] if hist["day_complete"].any() else hist
        for f in BEHAVIOR_FEATURES + GLUCOSE_FEATURES:
            row[f"hist_{f}"] = ref[f].mean()
        for day in INPUT_DAYS:
            row[f"d{day}_gl_mean"] = hist.loc[day, "gl_mean"]
        for day in EVAL_DAYS:
            for f in BEHAVIOR_FEATURES:
                row[f"d{day}_{f}"] = d.loc[(pid, day), f]
        row["target_d9_gl_mean"] = d.loc[(pid, TARGET_DAY), "gl_mean"]
        row["target_d9_tir"] = d.loc[(pid, TARGET_DAY), "tir"]
        row["target_d9_gl_max"] = d.loc[(pid, TARGET_DAY), "gl_max"]
        rows.append(row)
    wide = pd.DataFrame(rows)
    static = participants[["participant_id", "group_a1c", *[c for c in BIO_FEATURES if c in participants]]]
    return wide.merge(static, on="participant_id", how="left")


def build_day_level_dataset(daily: pd.DataFrame, participants: pd.DataFrame,
                            min_coverage: float = 0.8) -> pd.DataFrame:
    """Alternativa com MAIS linhas (≈ 8 por participante): comportamento do dia t + resumo do dia t-1 + bio
    -> glicose média do dia t. Útil porque só 45 linhas (tabela 'wide') é pouco para treinar uma MLP."""
    d = daily.sort_values(["participant_id", "day"]).copy()
    prev_cols = BEHAVIOR_FEATURES + GLUCOSE_FEATURES
    prev = d[["participant_id", "day", *prev_cols]].copy()
    prev["day"] += 1
    prev = prev.rename(columns={c: f"prev_{c}" for c in prev_cols})
    df = d.merge(prev, on=["participant_id", "day"], how="inner")
    df = df[(df["glucose_coverage"] >= min_coverage) & (df["day"] <= 10)]
    df = df[["participant_id", "day", "split", *BEHAVIOR_FEATURES, *[f"prev_{c}" for c in prev_cols], "gl_mean", "tir"]]
    df = df.rename(columns={"gl_mean": "target_gl_mean", "tir": "target_tir"})
    static = participants[["participant_id", "group_a1c", *[c for c in BIO_FEATURES if c in participants]]]
    return df.merge(static, on="participant_id", how="left")


# --------------------------------------------------------------------------------------
# 7. Divisão treino/validação/teste e escalonamento (sem vazamento)
# --------------------------------------------------------------------------------------
def split_participants(participants: pd.DataFrame, valid_size: float = 0.2, test_size: float = 0.2,
                       seed: int = 42, stratify_col: str = "group_a1c") -> dict[str, list[int]]:
    """Divide POR PARTICIPANTE (nunca por linha!), estratificando pelo grupo clínico.

    Dividir por linha misturaria dias do mesmo participante em treino e teste e inflaria o desempenho."""
    from sklearn.model_selection import train_test_split
    p = participants[participants["eligible"]] if "eligible" in participants else participants
    ids = p["participant_id"].to_numpy()
    strat = p[stratify_col].to_numpy() if stratify_col in p and p[stratify_col].notna().all() else None
    try:
        tv, test = train_test_split(ids, test_size=test_size, random_state=seed, stratify=strat)
        strat_tv = None if strat is None else p.set_index("participant_id").loc[tv, stratify_col].to_numpy()
        train, valid = train_test_split(tv, test_size=valid_size / (1 - test_size), random_state=seed,
                                        stratify=strat_tv)
    except ValueError:  # grupos pequenos demais para estratificar
        warnings.warn("Estratificação impossível; usando divisão aleatória simples.")
        tv, test = train_test_split(ids, test_size=test_size, random_state=seed)
        train, valid = train_test_split(tv, test_size=valid_size / (1 - test_size), random_state=seed)
    return {"train": sorted(int(i) for i in train), "valid": sorted(int(i) for i in valid),
            "test": sorted(int(i) for i in test)}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Colunas numéricas de entrada: exclui identificadores, alvos e colunas de controle."""
    skip = {"participant_id", "day", "split"}
    return [c for c in df.select_dtypes("number").columns if c not in skip and not c.startswith("target_")]


@dataclass
class FittedPreprocessor:
    """Imputer (mediana) + scaler ajustados SÓ com o treino. Use .transform() também em dados novos
    (por exemplo, os planos gerados pelo algoritmo genético)."""
    feature_cols: list[str]
    imputer: object
    scaler: object
    meta: dict = field(default_factory=dict)

    def transform(self, df: pd.DataFrame, clip: float | None = None) -> pd.DataFrame:
        """`clip` (opcional, para scaler='standard'): limita os valores escalados a [-clip, clip]. Protege a rede de
        entradas muito fora do treino (participantes atípicos; planos do AG que extrapolam o que o modelo viu)."""
        X = df.reindex(columns=self.feature_cols).astype(float)
        Z = self.scaler.transform(self.imputer.transform(X))
        if clip is not None:
            Z = np.clip(Z, -clip, clip)
        return pd.DataFrame(Z, columns=self.feature_cols, index=df.index)

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)


class _PreprocessorUnpickler(pickle.Unpickler):
    """Lê também arquivos gravados quando o pipeline rodou como script ('__main__.FittedPreprocessor')."""

    def find_class(self, module, name):
        if name == "FittedPreprocessor" and module in ("__main__", "preprocessing"):
            return FittedPreprocessor
        return super().find_class(module, name)


def load_preprocessor(path) -> FittedPreprocessor:
    with open(path, "rb") as fh:
        return _PreprocessorUnpickler(fh).load()


def fit_preprocessor(df: pd.DataFrame, splits: dict, feature_cols: list[str] | None = None,
                     scaler: str = "standard") -> FittedPreprocessor:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    train = df[df["participant_id"].isin(splits["train"])]
    candidates = list(feature_cols or get_feature_columns(df))
    all_nan = [c for c in candidates if not train[c].notna().any()]
    # Constante no treino = não carrega informação, e o scaler a deixaria em unidade bruta (escala 1): um valor
    # diferente em validação/teste (ou num plano do AG) viraria, por exemplo, z = 250 em vez de um valor padronizado.
    constant = [c for c in candidates if c not in all_nan
                and train[c].max() - train[c].min() <= 1e-9 * max(1.0, float(train[c].abs().max()))]
    cols = [c for c in candidates if c not in all_nan and c not in constant]
    imp = SimpleImputer(strategy="median").fit(train[cols])
    sc = (StandardScaler() if scaler == "standard" else MinMaxScaler()).fit(imp.transform(train[cols]))
    return FittedPreprocessor(cols, imp, sc, {"scaler": scaler, "n_train_rows": len(train),
                                              "dropped_constant": constant, "dropped_all_nan": all_nan})


def make_matrices(df: pd.DataFrame, target_col: str, splits: dict, prep: FittedPreprocessor) -> dict:
    """Devolve X_train/X_valid/X_test (escalados) e y_* (na unidade original, mg/dL ou %)."""
    out = {}
    for name in ("train", "valid", "test"):
        part = df[df["participant_id"].isin(splits[name])].dropna(subset=[target_col])
        out[f"X_{name}"] = prep.transform(part)
        out[f"y_{name}"] = part[target_col].astype(float)
        out[f"ids_{name}"] = part[["participant_id", *(["day"] if "day" in part else [])]]
    return out


# --------------------------------------------------------------------------------------
# 8. Pipeline completo
# --------------------------------------------------------------------------------------
def run_pipeline(raw_dir, out_dir, bio_path=None, glucose_source="consensus", day_mode="first_full",
                 min_coverage=0.8, seed=42, scaler="standard", verbose=True) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = find_participant_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"Nenhum 'CGMacros-XXX.csv' encontrado em {raw_dir.resolve()}")
    log = print if verbose else (lambda *a, **k: None)
    log(f"{len(files)} participantes encontrados")

    minutes, stats = [], []
    for pid, path in files.items():
        m, s = clean_participant(load_participant_csv(path), pid, glucose_source=glucose_source,
                                 day_mode=day_mode, min_coverage=min_coverage)
        minutes.append(m); stats.append(s)
    minute = pd.concat(minutes, ignore_index=True)
    stats = pd.DataFrame(stats)

    bio_path = Path(bio_path) if bio_path else find_bio_file(raw_dir)
    bio = load_bio(bio_path) if bio_path and Path(bio_path).exists() else None
    if bio is None:
        log("AVISO: bio.csv não encontrado; seguindo sem dados demográficos/laboratoriais.")

    fs_detail, fs_summary = (fingerstick_validation(minute, bio) if bio is not None
                             else (pd.DataFrame(), pd.DataFrame()))
    if len(fs_summary):
        log("Sensores x ponta de dedo (n=%d medições com os 2 sensores):" % int(fs_summary["n"].iloc[0]))
        log(fs_summary.round(1).to_string(index=False))
    log("Fonte de METs:", stats["mets_source"].value_counts().to_dict())

    daily = build_daily_features(minute, min_coverage)
    meals = extract_meals(minute)
    participants = build_participants_table(daily, stats, bio, min_coverage)
    wide = build_wide_dataset(daily, participants)
    day_level = build_day_level_dataset(daily, participants, min_coverage)
    splits = split_participants(participants, seed=seed)
    log(f"Elegíveis: {int(participants['eligible'].sum())}/{len(participants)} | "
        f"treino={len(splits['train'])} valid={len(splits['valid'])} teste={len(splits['test'])}")

    inp, ev = split_input_eval(daily)
    minute.to_csv(out_dir / "minute_clean.csv.gz", index=False)
    daily.to_csv(out_dir / "daily_features.csv", index=False)
    inp.to_csv(out_dir / "input_D1_D6.csv", index=False)
    ev.to_csv(out_dir / "eval_D7_D9.csv", index=False)
    meals.to_csv(out_dir / "meals.csv", index=False)
    participants.to_csv(out_dir / "participants.csv", index=False)
    stats.to_csv(out_dir / "quality_report.csv", index=False)
    if len(fs_detail):
        fs_detail.to_csv(out_dir / "fingerstick_validation.csv", index=False)
        fs_summary.to_csv(out_dir / "fingerstick_summary.csv", index=False)
    wide.to_csv(out_dir / "dataset_wide.csv", index=False)
    day_level.to_csv(out_dir / "dataset_day_level.csv", index=False)
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    config = {"glucose_source": glucose_source, "day_mode": day_mode, "min_coverage": min_coverage,
              "seed": seed, "scaler": scaler, "behavior_features": BEHAVIOR_FEATURES,
              "glucose_features": GLUCOSE_FEATURES, "bio_features": BIO_FEATURES}

    for name, df, target in (("wide", wide, "target_d9_gl_mean"), ("day_level", day_level, "target_gl_mean")):
        prep = fit_preprocessor(df, splits, scaler=scaler)
        mats = make_matrices(df, target, splits, prep)
        d = out_dir / "model_ready" / name
        d.mkdir(parents=True, exist_ok=True)
        for k, v in mats.items():
            (v.to_frame() if isinstance(v, pd.Series) else v).to_csv(d / f"{k}.csv", index=False)
        prep.save(d / "preprocessor.pkl")
        (d / "feature_columns.json").write_text(json.dumps(prep.feature_cols, indent=2))
        dropped = {"constant_in_train": prep.meta["dropped_constant"], "all_missing_in_train": prep.meta["dropped_all_nan"]}
        (d / "dropped_features.json").write_text(json.dumps(dropped, indent=2))
        config[f"{name}_target"] = target
        log(f"[{name}] {len(prep.feature_cols)} features ({len(prep.meta['dropped_constant'])} constantes no treino e "
            f"{len(prep.meta['dropped_all_nan'])} vazias descartadas) | linhas treino/valid/teste = "
            f"{len(mats['X_train'])}/{len(mats['X_valid'])}/{len(mats['X_test'])}")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    log(f"Arquivos salvos em {out_dir.resolve()}")
    return {"minute": minute, "daily": daily, "meals": meals, "participants": participants, "wide": wide,
            "day_level": day_level, "bio": bio, "splits": splits, "quality": stats,
            "fingerstick_detail": fs_detail, "fingerstick_summary": fs_summary}


def load_processed(out_dir="data/processed") -> dict[str, pd.DataFrame]:
    """Atalho para os outros integrantes carregarem a base já pronta."""
    out_dir = Path(out_dir)
    read = lambda n: pd.read_csv(out_dir / n)
    return {"daily": read("daily_features.csv"), "meals": read("meals.csv"),
            "participants": read("participants.csv"), "wide": read("dataset_wide.csv"),
            "day_level": read("dataset_day_level.csv"),
            "splits": json.loads((out_dir / "splits.json").read_text())}


def _cli():
    ap = argparse.ArgumentParser(description="Pré-processamento do CGMacros")
    ap.add_argument("--raw", default="data/raw", help="pasta com os CSVs do CGMacros (busca recursiva)")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--bio", default=None, help="caminho do bio.csv (padrão: procura dentro de --raw)")
    ap.add_argument("--glucose", default="consensus", choices=["consensus", "libre_filled", "mean", "libre", "dexcom"])
    ap.add_argument("--day-mode", default="first_full", choices=["first_full", "calendar", "from_start"])
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run_pipeline(a.raw, a.out, a.bio, a.glucose, a.day_mode, a.min_coverage, a.seed)


if __name__ == "__main__":
    # Roda a CLI a partir do módulo 'preprocessing' (e não de '__main__'): assim os .pkl gravados referenciam
    # 'preprocessing.FittedPreprocessor' e carregam nos módulos dos outros integrantes.
    import preprocessing as _module
    _module._cli()
