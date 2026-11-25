"""Gera um dataset SINTÉTICO no formato REAL do CGMacros, apenas para TESTAR o pipeline.

NÃO use estes dados para tirar conclusões: a fisiologia é inventada.
Reproduz as peculiaridades encontradas no dataset real (v1.0.0):
  - timestamps ISO ("2023-07-14 09:12:00"); um participante em formato americano (fallback do parser);
  - Libre já interpolado a cada minuto; Dexcom só começa ~75 min depois e tem buracos;
  - Libre lê ~25 mg/dL ABAIXO e Dexcom ~30 mg/dL ACIMA da glicose "verdadeira" (as pontas de dedo do bio.csv
    vêm da glicose verdadeira) -> a média dos dois é a melhor estimativa;
  - 11 participantes com `Intensity` (0/1) no lugar de `METs`;
  - cabeçalhos variados: coluna de índice "Unnamed: 0", Libre/Dexcom em ordem trocada, sem "Amount Consumed",
    com "Sugar", "Amount Consumed " com espaço no fim;
  - lanches escritos de 3 jeitos ("Snack", "Snacks", "snack 1"); fibra do jantar com erro de unidade (x150)
    em dois participantes; timestamps duplicados; leituras fora de 40-400;
  - participante 7 com só 7 dias (inelegível); bio.csv com colunas "Time (t)" repetidas, nomes com espaço e
    valores-erro (LDL=800, VLDL=400).

Uso:  python tools/make_synthetic_data.py data/synthetic_raw
Também grava `_synthetic_truth.json` (parâmetros verdadeiros) para os testes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

INTENSITY_IDS = {11, 26, 27, 31, 32, 33, 34, 35, 38, 42, 46}   # como no dataset real
INDEX_COL_IDS = {1, 2, 3, 4, 5, 6, 33, 36, 49}
SWAP_IDS, NO_AMOUNT_IDS, SUGAR_IDS, US_DATE_IDS = {4}, {28, 29}, {13}, {9}
FIBER_ERROR_IDS = {16, 19}
LIBRE_BIAS, DEXCOM_BIAS = -25.0, 30.0


def meal_response(n, start, carbs, fat, k=0.55):
    t = np.arange(n) - start
    out = np.zeros(n)
    m = t >= 0
    peak = 45 + 0.3 * fat
    out[m] = k * carbs * (t[m] / peak) * np.exp(1 - t[m] / peak)
    return out


def make_participant(pid, group, n_days=10):
    base = {"healthy": 92, "pre": 108, "t2d": 145}[group]
    start = pd.Timestamp("2020-05-04") + pd.Timedelta(days=int(rng.integers(0, 60)),
                                                       hours=int(rng.integers(8, 12)),
                                                       minutes=int(rng.integers(0, 60)))
    n = n_days * 1440 - int(rng.integers(0, 200))
    idx = pd.date_range(start, periods=n, freq="min")
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60

    glu = base + 4 * np.sin((hour - 4) / 24 * 2 * np.pi)
    meal_rows = []
    for d0 in pd.Series(idx.normalize()).unique():
        for mt, (h0, h1) in {"Breakfast": (7, 9.5), "Lunch": (11.5, 14), "Dinner": (18, 21)}.items():
            if rng.random() < 0.08:
                continue
            ts = pd.Timestamp(d0) + pd.Timedelta(hours=float(rng.uniform(h0, h1)))
            if ts < idx[0] + pd.Timedelta(hours=3) or ts > idx[-1] - pd.Timedelta(hours=3):
                continue
            carbs = float(rng.uniform(15, 120)); prot = float(rng.uniform(8, 50))
            fat = float(rng.uniform(5, 50)); fib = float(rng.uniform(1, 15))
            if pid in FIBER_ERROR_IDS and mt == "Dinner":
                fib *= 150                                           # erro de unidade nos jantares
            i = int((ts - idx[0]).total_seconds() // 60)
            glu = glu + meal_response(n, i, carbs, fat, k={"healthy": .35, "pre": .5, "t2d": .8}[group])
            meal_rows.append((i, mt, 4 * carbs + 4 * prot + 9 * fat, carbs, prot, fat, fib,
                              float(rng.choice([100, 100, 100, 75, 50]))))
        if rng.random() < 0.5:                                       # lanche
            ts = pd.Timestamp(d0) + pd.Timedelta(hours=float(rng.uniform(15, 17)))
            if idx[0] + pd.Timedelta(hours=3) < ts < idx[-1] - pd.Timedelta(hours=3):
                i = int((ts - idx[0]).total_seconds() // 60)
                meal_rows.append((i, str(rng.choice(["Snack", "Snacks", "snack 1"])), 200, 25, 6, 7, 2, 100.0))
    if pid == 3:                                                     # um erro de digitação isolado
        i, mt, cal, c, p, f, fi, amt = meal_rows[2]
        meal_rows[2] = (i, mt, cal, c, p, f, 2830, amt)

    # atividade: METs x 10, calorias = kcal_por_MET x METs
    mets10 = np.clip(np.round(10 + np.abs(rng.normal(0, 1.2, n))), 10, None)
    hr = 68 + rng.normal(0, 3, n)
    for d0 in pd.Series(idx.normalize()).unique():
        if rng.random() < 0.5:
            i = int((pd.Timestamp(d0) + pd.Timedelta(hours=float(rng.uniform(6, 19))) - idx[0]).total_seconds() // 60)
            if 0 <= i < n - 100:
                mets10[i:i + 40] += 35; hr[i:i + 40] += 50
                glu[i:i + 90] -= np.linspace(0, 15, 90)
    bmr_min = float(rng.uniform(0.8, 1.6))                           # kcal/min com 1 MET
    cal_act = mets10 / 10 * bmr_min

    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.97 * noise[t - 1] + rng.normal(0, 0.8)
    glu = np.clip(glu + noise, 50, 380)                              # glicose "verdadeira"

    # sensores: Libre (leitura a cada 15 min, interpolada) e Dexcom (5 min, começa ~75 min depois)
    jl, jd = rng.normal(0, 4), rng.normal(0, 4)
    minutes = np.arange(n)
    i15, i5 = minutes[minutes % 15 == 0], minutes[minutes % 5 == 0]
    libre = np.interp(minutes, i15, (glu + LIBRE_BIAS + jl + rng.normal(0, 3, n))[i15])
    dex = np.interp(minutes, i5, (glu + DEXCOM_BIAS + jd + rng.normal(0, 3, n))[i5])
    dex[:75] = np.nan
    for _ in range(4):
        s = int(rng.integers(100, n - 200)); L = int(rng.choice([20, 45, 120]))
        dex[s:s + L] = np.nan
        if rng.random() < .3:
            libre[s:s + L] = np.nan
    bad = rng.integers(200, n, 6)
    dex[bad[:3]] = 520; libre[bad[3:]] = 12

    cols = {"Timestamp": idx, "Libre GL": libre.round(2), "Dexcom GL": dex.round(2), "HR": hr.round(1),
            "Calories (Activity)": cal_act.round(5)}
    if pid in INTENSITY_IDS:
        cols["Intensity"] = (mets10 >= 24).astype(float)
    else:
        cols["METs"] = mets10
    meal_cols = ["Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber"] + \
                (["Sugar"] if pid in SUGAR_IDS else []) + \
                (["Amount Consumed" + (" " if pid == 1 else "")] if pid not in NO_AMOUNT_IDS else []) + ["Image path"]
    for c in meal_cols:
        cols[c] = pd.Series([None] * n, dtype=object) if c in ("Meal Type", "Image path") else np.nan
    df = pd.DataFrame(cols)
    amount_col = next((c for c in meal_cols if c.startswith("Amount")), None)
    for (i, mt, cal, c, p, f, fi, amt) in meal_rows:
        df.loc[i, "Meal Type"] = mt
        for name, v in (("Calories", cal), ("Carbs", c), ("Protein", p), ("Fat", f), ("Fiber", fi)):
            df.loc[i, name] = round(v)
        if amount_col:
            df.loc[i, amount_col] = amt
        df.loc[i, "Image path"] = f"photos/{pid:03d}-{i}.jpg"
    # relógio ausente por algumas horas
    s = int(rng.integers(0, n - 500))
    df.loc[s:s + 300, ["HR", "Calories (Activity)"] + (["Intensity"] if pid in INTENSITY_IDS else ["METs"])] = np.nan
    if pid in SWAP_IDS:
        order = list(df.columns); a, b = order.index("Libre GL"), order.index("Dexcom GL")
        order[a], order[b] = order[b], order[a]; df = df[order]
    dups = df.sample(5, random_state=pid)
    df = pd.concat([df, dups]).sort_values("Timestamp", kind="stable").reset_index(drop=True)
    fmt = "%m/%d/%Y %H:%M" if pid in US_DATE_IDS else "%Y-%m-%d %H:%M:%S"
    df["Timestamp"] = df["Timestamp"].dt.strftime(fmt)
    if pid in INDEX_COL_IDS:
        df.insert(0, "Unnamed: 0", np.arange(len(df)))
    return df, start, glu, {"kcal_per_met": bmr_min, "intensity_file": pid in INTENSITY_IDS}


def main(out):
    out = Path(out)
    ids = [i for i in range(1, 50) if i not in (24, 25, 37, 40)]     # 45 participantes, como no dataset real
    groups = ["healthy"] * 15 + ["pre"] * 16 + ["t2d"] * 14
    rng.shuffle(groups)
    rows, truth = [], {}
    for pid, g in zip(ids, groups):
        d, start, glu, info = make_participant(pid, g, 7 if pid == 7 else 10)
        f = out / f"CGMacros-{pid:03d}" / f"CGMacros-{pid:03d}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(f, index=False)
        truth[pid] = info
        a1c = {"healthy": rng.normal(5.2, .2), "pre": rng.normal(6.0, .2), "t2d": rng.normal(7.2, .6)}[g]
        glu0 = {"healthy": 88, "pre": 104, "t2d": 140}[g] + rng.normal(0, 6)
        ins = {"healthy": 6, "pre": 12, "t2d": 20}[g] + rng.normal(0, 2)
        fs = []   # pontas de dedo = glicose verdadeira: #1 antes do início do registro, #2 e #3 depois
        for off in (-15, 90, 150):
            t = start + pd.Timedelta(minutes=off)
            fs += [round(float(glu[max(off, 0)])), f"{t.hour}:{t.minute:02d}"]
        rows.append([pid, int(rng.integers(18, 69)), rng.choice(["F", "M"]), rng.uniform(21, 48),
                     rng.uniform(117, 280), rng.integers(59, 73),
                     rng.choice(["African American Black", "African American", "Hispanic/Latino", "White"]),
                     a1c, glu0, ins, rng.uniform(40, 300), rng.uniform(100, 300), rng.uniform(25, 100),
                     rng.uniform(50, 200), 800 if pid == 12 else rng.uniform(40, 180), 400 if pid == 12 else rng.uniform(8, 60),
                     rng.uniform(2, 6), "7:34:00 AM", *fs])
    header = ("subject,Age,Gender,BMI,Body weight ,Height ,Self-identify ,A1c PDL (Lab),Fasting GLU - PDL (Lab),"
              "Insulin ,Triglycerides,Cholesterol,HDL,Non HDL ,LDL (Cal),VLDL (Cal),Cho/HDL Ratio,"
              "Collection time PDL (Lab),#1 Contour Fingerstick GLU,Time (t), #2 Contour Fingerstick GLU,Time (t),"
              "#3 Contour Fingerstick GLU,Time (t)")
    with open(out / "bio.csv", "w") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(str(round(x, 2)) if isinstance(x, (float, np.floating)) else str(x) for x in r) + "\n")
    (out / "_synthetic_truth.json").write_text(json.dumps({str(k): v for k, v in truth.items()}))
    print(f"Dataset sintético gerado em {out} ({len(ids)} participantes)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_raw")
