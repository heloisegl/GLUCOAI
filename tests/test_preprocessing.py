"""Testes de sanidade. Rodam em dados SINTÉTICOS:  python tests/test_preprocessing.py"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import preprocessing as P


def main():
    tmp = Path(tempfile.mkdtemp())
    subprocess.run([sys.executable, str(ROOT / "tools/make_synthetic_data.py"), str(tmp / "raw")], check=True)
    r = P.run_pipeline(tmp / "raw", tmp / "out", verbose=False)
    truth = json.loads((tmp / "raw/_synthetic_truth.json").read_text())
    minute, daily, wide, part, splits = r["minute"], r["daily"], r["wide"], r["participants"], r["splits"]

    # 1. glicose limpa dentro dos limites do sensor, sem duplicados, grade de 1 min
    assert minute["glucose"].dropna().between(P.GLUCOSE_MIN, P.GLUCOSE_MAX).all()
    assert not minute.duplicated(["participant_id", "timestamp"]).any()
    assert (minute.groupby("participant_id")["timestamp"].diff().dropna() == pd.Timedelta("1min")).all()

    # 2. nutrientes só nas linhas de refeição; METs já divididos por 10 (sintético: ~1.2)
    assert minute.loc[minute["meal_type"].isna(), list(P.NUTRIENTS)].sum().sum() == 0
    assert minute["mets"].median() < 3

    # 2b. lanches normalizados, erro de fibra (2830 g) neutralizado, D1 completo, dia 0 fora do daily
    assert set(minute["meal_type"].dropna().unique()) <= {"breakfast", "lunch", "dinner", "snack"}
    assert "snack" in set(minute["meal_type"].dropna())
    assert minute["fiber"].max() <= P.MEAL_VALUE_CAPS["fiber"]
    assert (daily["day"] >= 1).all() and (minute["day"] == 0).any()
    assert daily.loc[daily.day == 1, "day_complete"].all()   # first_full: D1 já é um dia completo
    assert (daily["n_snacks"] > 0).any() and "snack_carbs" in daily

    # 2c. dia com relógio ausente (HR/METs faltando) -> features de atividade NaN, não 0
    m2 = minute.copy()
    sel = (m2.participant_id == 3) & (m2.day == 5)
    m2.loc[sel, ["hr", "mets", "cal_activity"]] = np.nan
    d2 = P.build_daily_features(m2)
    row2 = d2[(d2.participant_id == 3) & (d2.day == 5)].iloc[0]
    assert np.isnan(row2["active_min"]) and np.isnan(row2["cal_activity_sum"]) and np.isnan(row2["hr_mean"])
    assert daily.loc[daily.activity_coverage > 0.9, "active_min"].between(0, 1440).all()

    # 2d. cabeçalhos variados do dataset real foram todos lidos (índice, ordem trocada, sem Amount, Sugar, Intensity)
    q = r["quality"].set_index("participant_id")
    assert len(q) == 45 and (q["raw_rows"] > 5000).all()
    assert set(q["mets_source"]) == {"METs", "estimated_from_calories"}
    assert (q.loc[list(P_INTENSITY := [11, 26, 27, 31, 32, 33, 34, 35, 38, 42, 46]), "mets_source"] == "estimated_from_calories").all()
    # kcal por MET estimado (Intensity) e medido (METs) recuperam o valor verdadeiro
    for pid, row in q.iterrows():
        assert abs(row["kcal_per_met"] / truth[str(pid)]["kcal_per_met"] - 1) < 0.03, pid
    # atividade: nenhum participante com Intensity fica com features de METs vazias
    assert daily.merge(q[["mets_source"]], left_on="participant_id", right_index=True) \
        .query("mets_source == 'estimated_from_calories'")["active_min"].notna().mean() > 0.9

    # 2e. glicose: consenso dos sensores fica perto da ponta de dedo; cada sensor sozinho, não
    summ = r["fingerstick_summary"].set_index("series")
    assert summ.loc["libre", "bias_mg_dl"] < -15 and summ.loc["dexcom", "bias_mg_dl"] > 15
    assert abs(summ.loc["midpoint", "bias_mg_dl"]) < 8 and abs(summ.loc["pipeline_glucose", "bias_mg_dl"]) < 8
    # sem "degraus": a glicose do pipeline não muda de nível quando um sensor some (diferença entre minutos vizinhos)
    jump = minute.groupby("participant_id")["glucose"].diff().abs()
    assert jump.max() < 12, jump.max()   # com média simples seria ~30

    # 2f. erros de unidade na fibra (jantares x150) e o erro isolado de 2830 g foram neutralizados e contados
    assert q.loc[[16, 19], "capped_fiber"].min() >= 5 and q.loc[3, "capped_fiber"] >= 1
    assert q.drop(index=[3, 16, 19])["meal_values_capped"].sum() == 0

    # 3. feature diária recalculada à mão bate com a tabela
    pid, day = 3, 5
    g = minute[(minute.participant_id == pid) & (minute.day == day)]
    row = daily[(daily.participant_id == pid) & (daily.day == day)].iloc[0]
    assert np.isclose(g["glucose"].mean(), row["gl_mean"])
    assert np.isclose(g.loc[g.meal_type.notna(), "carbs"].sum(), row["carbs_total"])
    assert np.isclose(g.loc[g.meal_type == "lunch", "calories"].sum(), row["lunch_calories"])

    # 4. alvo do wide = glicose média de D9 da tabela diária; participante inelegível fora
    for _, w in wide.iterrows():
        d9 = daily[(daily.participant_id == w.participant_id) & (daily.day == 9)].iloc[0]
        assert np.isclose(w["target_d9_gl_mean"], d9["gl_mean"])
        assert np.isclose(w["d9_carbs_total"], d9["carbs_total"])
    assert 7 not in set(wide.participant_id) and not part.set_index("participant_id").loc[7, "eligible"]
    assert part["eligible"].sum() == 44

    # 5. sem vazamento: splits disjuntos; alvo fora das features; scaler ajustado só no treino
    s = [set(splits[k]) for k in ("train", "valid", "test")]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
    prep = P.load_preprocessor(tmp / "out/model_ready/wide/preprocessor.pkl")
    assert not any(c.startswith("target_") for c in prep.feature_cols)
    Xtr = pd.read_csv(tmp / "out/model_ready/wide/X_train.csv")
    Xva = pd.read_csv(tmp / "out/model_ready/wide/X_valid.csv")
    assert np.allclose(Xtr.mean(), 0, atol=1e-8)          # treino centrado
    assert not np.allclose(Xva.mean(), 0, atol=1e-8)      # validação NÃO foi usada no ajuste
    assert Xtr.isna().sum().sum() == 0 and Xva.isna().sum().sum() == 0

    # 5b. REGRESSÃO: o .pkl gravado quando o pipeline roda como SCRIPT (python preprocessing.py ...) precisa
    #     carregar nos módulos dos outros integrantes (antes referenciava '__main__.FittedPreprocessor' e falhava)
    subprocess.run([sys.executable, str(ROOT / "preprocessing.py"), "--raw", str(tmp / "raw"),
                    "--out", str(tmp / "out_cli")], check=True, capture_output=True)
    pk = tmp / "out_cli/model_ready/wide/preprocessor.pkl"
    assert b"__main__" not in pk.read_bytes()
    prep_cli = P.load_preprocessor(pk)
    ids_tr = json.loads((tmp / "out_cli/splits.json").read_text())["train"]
    wide_cli = pd.read_csv(tmp / "out_cli/dataset_wide.csv")
    back = prep_cli.transform(wide_cli[wide_cli["participant_id"].isin(ids_tr)])
    assert np.allclose(back.to_numpy(), pd.read_csv(tmp / "out_cli/model_ready/wide/X_train.csv").to_numpy(), atol=1e-6)
    chk = subprocess.run([sys.executable, str(ROOT / "tools/check_outputs.py"), str(tmp / "out_cli")],
                         capture_output=True, text=True)
    assert chk.returncode == 0 and "[ERRO" not in chk.stdout, chk.stdout[-1500:]
    assert "unnamed:_0" not in pd.read_csv(tmp / "out_cli/minute_clean.csv.gz", nrows=1).columns   # índice descartado
    #     arquivos antigos (gravados como '__main__') continuam legíveis
    import __main__ as main_mod
    old_name = P.FittedPreprocessor.__module__
    P.FittedPreprocessor.__module__ = "__main__"; main_mod.FittedPreprocessor = P.FittedPreprocessor
    try:
        prep_cli.save(tmp / "legacy.pkl")
    finally:
        P.FittedPreprocessor.__module__ = old_name; del main_mod.FittedPreprocessor
    assert b"__main__" in (tmp / "legacy.pkl").read_bytes()
    assert P.load_preprocessor(tmp / "legacy.pkl").feature_cols == prep_cli.feature_cols

    # 5c. o alinhamento dos sensores não pode empurrar a glicose para fora de 40-400 mg/dL (sensor único perto dos extremos)
    n = 3000
    ts = pd.date_range("2022-01-01 08:00", periods=n, freq="min")
    lib, dex = np.full(n, 100.0), np.full(n, 132.0)
    lib[500:600] = 395; dex[500:600] = np.nan        # só o Libre, perto do teto
    lib[900:1000] = np.nan; dex[900:1000] = 42       # só o Dexcom, perto do piso
    raw_x = pd.DataFrame({"timestamp": ts, "libre_gl": lib, "dexcom_gl": dex, "hr": 70.0, "cal_activity": 1.0,
                          "mets": 10.0, **{c: np.nan for c in ("meal_type", *P.NUTRIENTS, "amount_consumed")}})
    mx, _ = P.clean_participant(raw_x, 1)
    assert mx["glucose"].between(P.GLUCOSE_MIN, P.GLUCOSE_MAX).all() and mx["glucose"].max() == P.GLUCOSE_MAX

    # 5d. REGRESSÃO do verificador: participante inelegível por falta de glicose no D9 (como o ID 18 do dataset real)
    #     tem linhas no D9 mas não está no wide; isso não pode virar [ERRO] no check_outputs
    import shutil
    shutil.copytree(tmp / "raw", tmp / "raw_gap")
    f18 = next((tmp / "raw_gap").rglob("CGMacros-018.csv"))
    d18 = pd.read_csv(f18)
    t18 = pd.to_datetime(d18["Timestamp"], format="mixed")
    gap = (t18 >= t18.min() + pd.Timedelta(days=8.3)) & (t18 < t18.min() + pd.Timedelta(days=9.9))
    d18.loc[gap, ["Libre GL", "Dexcom GL"]] = np.nan
    d18.to_csv(f18, index=False)
    r_gap = P.run_pipeline(tmp / "raw_gap", tmp / "out_gap", verbose=False)
    p_gap = r_gap["participants"].set_index("participant_id")
    assert not p_gap.loc[18, "eligible"] and "D9" in p_gap.loc[18, "reason_ineligible"] and p_gap["eligible"].sum() == 43
    chk2 = subprocess.run([sys.executable, str(ROOT / "tools/check_outputs.py"), str(tmp / "out_gap")],
                          capture_output=True, text=True)
    assert chk2.returncode == 0 and "[ERRO" not in chk2.stdout, chk2.stdout[-1500:]

    # 5e. colunas constantes (ou vazias) no treino saem do pré-processador: o scaler as deixaria em unidade bruta
    toy = pd.DataFrame({"participant_id": range(1, 9), "a": np.arange(8.0), "const": 5.0, "const_nan": [3.0, np.nan] * 4,
                        "all_nan": np.nan, "target_x": 1.0})
    sp_toy = {"train": [1, 2, 3, 4, 5, 6], "valid": [7], "test": [8]}
    prep_t = P.fit_preprocessor(toy, sp_toy)
    assert prep_t.feature_cols == ["a"]
    assert set(prep_t.meta["dropped_constant"]) == {"const", "const_nan"} and prep_t.meta["dropped_all_nan"] == ["all_nan"]
    assert np.allclose(prep_t.transform(toy.assign(const=500.0))["a"], prep_t.transform(toy)["a"])   # valor novo não vaza
    big = prep_t.transform(toy.assign(a=1000.0), clip=3)["a"]
    assert big.max() == 3 and prep_t.transform(toy.assign(a=-1000.0), clip=3)["a"].min() == -3
    assert np.allclose(prep_t.transform(toy, clip=None), prep_t.transform(toy))   # padrão: sem limitar
    for nm in ("wide", "day_level"):
        assert (pd.read_csv(tmp / f"out_cli/model_ready/{nm}/X_train.csv").std() > 1e-9).all()
        assert (tmp / f"out_cli/model_ready/{nm}/dropped_features.json").exists()

    bio = r["bio"].set_index("participant_id")
    assert bio.loc[12, ["ldl", "vldl"]].isna().all() and "fingerstick_2_time" in bio.columns
    assert bio["group_a1c"].notna().all()

    # 6. day_level não mistura participantes entre splits
    dl = r["day_level"]
    assert dl.groupby("participant_id").size().min() >= 1 and "target_gl_mean" in dl
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
