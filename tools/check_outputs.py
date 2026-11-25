"""Confere a base gerada em data/processed (rode DEPOIS do pipeline).

    python tools/check_outputs.py                # confere data/processed
    python tools/check_outputs.py caminho/da/pasta

Cada linha é [OK], [INFO], [AVISO] (vale olhar, pode estar certo) ou [ERRO] (algo está errado).
Sai com código 1 se houver algum [ERRO]. Não precisa do dataset original, só das saídas.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import preprocessing as P  # noqa: E402

LEAK = re.compile(r"^(target_|d[789]_(gl_|tir|tar|tbr))")   # glicose de D7-D9 / alvos NÃO podem ser entrada
counts = {"OK": 0, "INFO": 0, "AVISO": 0, "ERRO": 0}


def say(level, msg):
    counts[level] += 1
    print(f"[{level:5s}] {msg}")


def check(cond, msg, level="ERRO", detail=""):
    if cond:
        say("OK", msg)
    else:
        say(level, msg + (f"  ->  {detail}" if detail else ""))
    return bool(cond)


def main(out_dir):
    out = Path(out_dir)
    print(f"Conferindo: {out.resolve()}\n")

    # ---------------------------------------------------------------- arquivos
    needed = ["minute_clean.csv.gz", "daily_features.csv", "input_D1_D6.csv", "eval_D7_D9.csv", "meals.csv",
              "participants.csv", "dataset_wide.csv", "dataset_day_level.csv", "splits.json",
              "quality_report.csv", "config.json"]
    for name in ("wide", "day_level"):
        needed += [f"model_ready/{name}/{f}" for f in
                   ("X_train.csv", "X_valid.csv", "X_test.csv", "y_train.csv", "y_valid.csv", "y_test.csv",
                    "ids_train.csv", "ids_valid.csv", "ids_test.csv", "preprocessor.pkl", "feature_columns.json")]
    missing = [f for f in needed if not (out / f).exists()]
    if not check(not missing, "Todos os arquivos esperados existem", detail=f"faltam: {missing}"):
        print("\nRode o pipeline antes:  python preprocessing.py --raw <pasta do dataset> --out data/processed")
        return 1
    check((out / "fingerstick_summary.csv").exists(), "Validação com ponta de dedo gravada", "AVISO",
          "bio.csv não encontrado ou sem pontas de dedo")

    minute = pd.read_csv(out / "minute_clean.csv.gz", parse_dates=["timestamp"])
    daily = pd.read_csv(out / "daily_features.csv")
    part = pd.read_csv(out / "participants.csv")
    wide = pd.read_csv(out / "dataset_wide.csv")
    dlev = pd.read_csv(out / "dataset_day_level.csv")
    quality = pd.read_csv(out / "quality_report.csv")
    splits = json.loads((out / "splits.json").read_text())
    config = json.loads((out / "config.json").read_text())
    elig = set(part.loc[part["eligible"], "participant_id"])

    say("INFO", f"Configuração: glicose={config['glucose_source']} | dias={config['day_mode']} | "
                f"cobertura mínima={config['min_coverage']} | seed={config['seed']}")

    # ---------------------------------------------------------------- série por minuto
    print("\n-- Série por minuto")
    g = minute["glucose"].dropna()
    check(g.between(P.GLUCOSE_MIN, P.GLUCOSE_MAX).all(), f"Glicose dentro de {P.GLUCOSE_MIN}-{P.GLUCOSE_MAX} mg/dL",
          detail=f"min={g.min():.1f} max={g.max():.1f}")
    check(not minute.duplicated(["participant_id", "timestamp"]).any(), "Sem timestamps duplicados por participante")
    step = minute.sort_values(["participant_id", "timestamp"]).groupby("participant_id")["timestamp"].diff().dropna()
    check((step == pd.Timedelta("1min")).all(), "Grade regular de 1 minuto")
    miss = minute.groupby("participant_id")["glucose"].apply(lambda s: s.isna().mean())
    check(miss.mean() < 0.05, f"Glicose faltante no total: {100 * miss.mean():.1f}%", "AVISO", "acima de 5%")
    bad = miss[(miss > 0.20) & miss.index.isin(elig)]
    check(bad.empty, "Nenhum participante elegível com > 20% de glicose faltante", "AVISO", f"IDs: {bad.index.tolist()}")
    gaps = miss[(miss > 0.20) & ~miss.index.isin(elig)]
    if len(gaps):
        say("INFO", f"Muita glicose faltante só em inelegíveis (esperado): IDs {gaps.index.tolist()}")
    say("INFO", f"{minute['participant_id'].nunique()} participantes | {len(minute):,} minutos | "
                f"glicose média geral = {g.mean():.1f} mg/dL")
    if {"libre_gl", "dexcom_gl"} <= set(minute):
        both = minute[["libre_gl", "dexcom_gl"]].dropna()
        say("INFO", f"Libre x Dexcom: Dexcom - Libre = {(both.dexcom_gl - both.libre_gl).mean():+.1f} mg/dL, "
                    f"correlação {both.corr().iloc[0, 1]:.2f}")

    # ---------------------------------------------------------------- dias e elegibilidade
    print("\n-- Dias e elegibilidade")
    check(daily["gl_mean"].dropna().between(P.GLUCOSE_MIN, P.GLUCOSE_MAX).all(), "gl_mean diário dentro dos limites")
    need = set(range(1, P.TARGET_DAY + 1))
    d = daily[daily["participant_id"].isin(elig)]
    days_ok = d.groupby("participant_id")["day"].apply(lambda s: need <= set(s))
    check(days_ok.all(), f"Todos os {len(elig)} elegíveis têm D1..D9", detail=f"IDs: {days_ok[~days_ok].index.tolist()}")
    d9 = d[d["day"] == P.TARGET_DAY]
    check((d9["glucose_coverage"] >= config["min_coverage"]).all(), "D9 dos elegíveis tem glicose suficiente")
    first = d[d["day"].between(1, 9)]
    say("INFO", f"Dias completos entre D1-D9 dos elegíveis: {100 * first['day_complete'].mean():.0f}%")
    inel = part.loc[~part["eligible"], ["participant_id", "reason_ineligible"]]
    say("INFO", f"Elegíveis: {len(elig)} de {len(part)} | inelegíveis: "
                + "; ".join(f"ID {r.participant_id} ({r.reason_ineligible})" for r in inel.itertuples()))
    check(len(elig) >= 40, "Pelo menos 40 participantes elegíveis", "AVISO", f"só {len(elig)}")
    big = d[(d["cal_total"] > 6000) | (d["carbs_total"] > 800)]
    check(big.empty, "Totais diários de calorias/carboidratos plausíveis", "AVISO",
          f"{len(big)} dias suspeitos, ex.: {big[['participant_id', 'day', 'cal_total', 'carbs_total']].head(3).to_dict('records')}")

    # ---------------------------------------------------------------- tabela wide e consistência com a série
    print("\n-- Tabela wide (1 linha por elegível)")
    check(set(wide["participant_id"]) == elig and wide["participant_id"].is_unique,
          "wide tem exatamente 1 linha por participante elegível")
    t = wide["target_d9_gl_mean"]
    check(t.notna().all(), "Alvo (glicose média em D9) sem faltantes")
    check(t.between(P.GLUCOSE_MIN, P.GLUCOSE_MAX).all() and t.between(70, 250).mean() > 0.9,
          f"Alvo plausível: média {t.mean():.1f}, mín {t.min():.1f}, máx {t.max():.1f} mg/dL", "AVISO")
    # recalcula o alvo direto da série por minuto (D9) e compara
    m9 = minute[minute["day"] == P.TARGET_DAY].groupby("participant_id")["glucose"].mean()
    tgt = wide.set_index("participant_id")["target_d9_gl_mean"]
    diff = (tgt - m9.reindex(tgt.index)).abs()          # só os participantes do wide (os inelegíveis ficam de fora)
    check(diff.notna().all() and diff.max() < 1e-6, "Alvo bate com a média da glicose do D9 recalculada da série por minuto",
          detail=f"maior diferença {diff.max():.4f}; sem valor recalculado para {diff[diff.isna()].index.tolist()}")
    hist_r = wide["hist_gl_mean"].corr(t)
    say("INFO", f"Correlação entre glicose média de D1-D6 e o alvo em D9: {hist_r:.2f}")
    nan_frac = wide.select_dtypes("number").isna().mean()
    bad = nan_frac[nan_frac > 0.5]
    check(bad.empty, "Nenhuma coluna do wide com > 50% de faltantes", "AVISO", f"{bad.round(2).to_dict()}")

    # ---------------------------------------------------------------- divisão
    print("\n-- Divisão treino / validação / teste")
    s = {k: set(v) for k, v in splits.items()}
    check(not (s["train"] & s["valid"]) and not (s["train"] & s["test"]) and not (s["valid"] & s["test"]),
          "Conjuntos disjuntos (nenhum participante em dois conjuntos)")
    check(s["train"] | s["valid"] | s["test"] == elig, "A união dos conjuntos é exatamente o conjunto de elegíveis")
    grp = part.set_index("participant_id")["group_a1c"]
    tab = pd.DataFrame({k: grp.loc[sorted(v)].value_counts() for k, v in s.items()}).fillna(0).astype(int)
    say("INFO", "Participantes por grupo (A1c) em cada conjunto:\n" + tab.to_string())
    check((tab > 0).all().all(), "Os três grupos clínicos aparecem em todos os conjuntos", "AVISO")

    # ---------------------------------------------------------------- matrizes prontas e pré-processador
    for name, df, target, key in (("wide", wide, "target_d9_gl_mean", ["participant_id"]),
                                  ("day_level", dlev, "target_gl_mean", ["participant_id", "day"])):
        print(f"\n-- model_ready/{name}")
        md = out / "model_ready" / name
        cols = json.loads((md / "feature_columns.json").read_text())
        X = {k: pd.read_csv(md / f"X_{k}.csv") for k in ("train", "valid", "test")}
        y = {k: pd.read_csv(md / f"y_{k}.csv").iloc[:, 0] for k in ("train", "valid", "test")}
        ids = {k: pd.read_csv(md / f"ids_{k}.csv") for k in ("train", "valid", "test")}
        say("INFO", f"{len(cols)} features | linhas treino/valid/teste = {len(X['train'])}/{len(X['valid'])}/{len(X['test'])}")
        check(all(list(X[k].columns) == cols for k in X), "X_* têm as colunas de feature_columns.json, na mesma ordem")
        check(all(len(X[k]) == len(y[k]) == len(ids[k]) for k in X), "X, y e ids têm o mesmo número de linhas")
        check(all(np.isfinite(X[k].to_numpy()).all() for k in X), "X sem NaN/infinito (imputação funcionou)")
        check(np.allclose(X["train"].mean(), 0, atol=1e-6), "X_train centrado em 0 (scaler ajustado no treino)")
        check(all(set(ids[k]["participant_id"]) <= s[k] for k in X), "Cada linha está no conjunto certo (sem vazamento entre participantes)")
        leak = [c for c in cols if LEAK.match(c)]
        check(not leak, "Nenhuma feature usa o alvo ou a glicose de D7-D9", detail=f"{leak}")
        check("participant_id" not in cols and "day" not in cols, "IDs e dia não entram como feature")
        keys = ["participant_id"] + (["day"] if "day" in ids["train"] else [])
        dfi = df.set_index(keys)
        tr = df[df["participant_id"].isin(s["train"])]
        lines = []
        for k in ("valid", "test"):
            for r, c in zip(*np.where(X[k].abs().to_numpy() > 10)):
                kv = tuple(ids[k].iloc[r][keys])
                raw_v = dfi.loc[kv[0] if len(kv) == 1 else kv, cols[c]]
                raw_v = raw_v.iloc[0] if hasattr(raw_v, "iloc") else raw_v
                lines.append(f"{'/'.join(str(v) for v in kv)} ({k}): {cols[c]} = {raw_v:.4g} "
                             f"(treino: média {tr[cols[c]].mean():.4g}, desvio {tr[cols[c]].std():.4g}; z = {X[k].iloc[r, c]:.1f})")
        check(not lines, "Sem valores extremos (|z| > 10) em validação/teste", "AVISO",
              "participante[/dia] (conjunto): feature = valor bruto\n          " + "\n          ".join(lines[:12]))
        const = [c for c in cols if X["train"][c].std() < 1e-9]
        check(not const, "Nenhuma feature constante no treino entra no modelo", "AVISO",
              f"{const} (rode o pipeline de novo com o preprocessing.py mais recente)")
        dj = md / "dropped_features.json"
        if dj.exists():
            dc = json.loads(dj.read_text()).get("constant_in_train", [])
            say("INFO", f"Descartadas por serem constantes no treino: {len(dc)}" + (f"\n          {dc}" if dc else ""))
            vt = df[df["participant_id"].isin(s["valid"] | s["test"])]
            moved = {}
            for c in dc:
                ref, other = tr[c].dropna(), vt[c].dropna()
                if ref.empty:
                    continue
                who = vt.loc[other.index[(other - ref.iloc[0]).abs() > 1e-6], "participant_id"].unique().tolist()
                if who:
                    moved[c] = who
            if moved:
                say("INFO", "Descartadas, mas com valor diferente do treino em validação/teste (ficam fora do modelo): "
                            + "; ".join(f"{c} (IDs {v})" for c, v in list(moved.items())[:8]) + (" ..." if len(moved) > 8 else ""))
        else:
            say("AVISO", "dropped_features.json não existe (saída gerada por versão antiga do pipeline)")
        # o preprocessor salvo reproduz X_train a partir do CSV bruto (é o que os Integrantes 2 e 3 vão usar)
        try:
            prep = P.load_preprocessor(md / "preprocessor.pkl")
            sub = df[df["participant_id"].isin(s["train"])].dropna(subset=[target])
            back = prep.transform(sub)
            check(prep.feature_cols == cols and np.allclose(back.to_numpy(), X["train"].to_numpy(), atol=1e-6),
                  "preprocessor.pkl carrega e reproduz X_train (use-o em dados novos / no AG)")
        except Exception as e:  # noqa: BLE001
            check(False, "preprocessor.pkl carrega e reproduz X_train", detail=f"{type(e).__name__}: {e}")

    # ---------------------------------------------------------------- sensores e qualidade
    print("\n-- Sensores e qualidade")
    fs = out / "fingerstick_summary.csv"
    if fs.exists():
        f = pd.read_csv(fs).set_index("series")
        say("INFO", f"Sensores x ponta de dedo (n={int(f['n'].iloc[0])}):\n" + f.drop(columns="n").round(1).to_string())
        pipe = f.loc["pipeline_glucose"]
        single = f.loc[["libre", "dexcom"]]
        check(abs(pipe["bias_mg_dl"]) <= single["bias_mg_dl"].abs().min() and pipe["mae_mg_dl"] <= single["mae_mg_dl"].min(),
              "A glicose usada no pipeline erra menos que qualquer sensor sozinho (viés e MAE)", "AVISO",
              "reveja a opção --glucose")
    off = quality["dexcom_minus_libre_median"].dropna()
    say("INFO", f"Deslocamento mediano Dexcom - Libre por participante: mediana {off.median():.1f} "
                f"(mín {off.min():.1f}, máx {off.max():.1f}) mg/dL")
    neg = quality.loc[quality["dexcom_minus_libre_median"] < 0, ["participant_id", "dexcom_minus_libre_median"]]
    if len(neg):
        say("INFO", "Participantes em que o Libre lê MAIS que o Dexcom (ID: deslocamento): "
                    + ", ".join(f"{r.participant_id}: {r.dexcom_minus_libre_median:.1f}" for r in neg.itertuples()))
    check(off.abs().max() < 90, "Nenhum deslocamento Dexcom - Libre absurdo (< 90 mg/dL)", "AVISO",
          f"IDs: {quality.loc[quality['dexcom_minus_libre_median'].abs() >= 90, 'participant_id'].tolist()}")
    say("INFO", f"Fonte dos METs: {quality['mets_source'].value_counts().to_dict()}")
    check((quality["mets_source"] != "missing").all(), "Todos os participantes têm METs (do arquivo ou estimados)", "AVISO",
          f"IDs: {quality.loc[quality['mets_source'] == 'missing', 'participant_id'].tolist()}")
    capped = [c for c in quality.columns if c.startswith("capped_")]
    tot = quality[capped].sum()
    say("INFO", "Macros acima do teto (viraram faltantes e foram imputados): " + ", ".join(f"{c[7:]}={int(v)}" for c, v in tot.items()))
    heavy = quality.loc[quality[capped].sum(axis=1) > 3, "participant_id"].tolist()
    check(not heavy, "Nenhum participante com mais de 3 valores de macros corrigidos", "AVISO", f"IDs: {heavy}")
    low = quality.loc[(quality["n_meals"] < 25) & quality["participant_id"].isin(elig), ["participant_id", "n_meals"]]
    check(low.empty, "Todos os elegíveis com pelo menos 25 refeições registradas", "AVISO",
          f"possível sub-registro (ID: refeições): {dict(zip(low.participant_id, low.n_meals))}")

    print(f"\nRESUMO: {counts['OK']} ok | {counts['AVISO']} aviso(s) | {counts['ERRO']} erro(s)")
    print("RESULTADO: " + ("TUDO CERTO" if counts["ERRO"] == 0 and counts["AVISO"] == 0 else
                           "SEM ERROS (confira os avisos acima)" if counts["ERRO"] == 0 else "HÁ ERROS, veja as linhas [ERRO]"))
    return 1 if counts["ERRO"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "processed"))
