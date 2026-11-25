"""Testes de sanidade da rede neural, em dados SINTÉTICOS: python tests/test_model.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model as M


def main():
    rng = np.random.default_rng(0)

    # 1. a rede consegue aprender uma relação linear simples (sanity check do loop de treino)
    n, p = 200, 5
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    true_w = rng.normal(size=p)
    y = pd.Series(X.to_numpy() @ true_w + rng.normal(scale=0.05, size=n))
    X_train, X_valid = X.iloc[:150], X.iloc[150:]
    y_train, y_valid = y.iloc[:150], y.iloc[150:]

    result = M.train_model(X_train, y_train, X_valid, y_valid, hidden_sizes=(16, 8),
                            dropout=0.0, max_epochs=300, patience=50, verbose=False)
    metrics = M.evaluate(result.model, X_valid, y_valid)
    assert metrics["r2"] > 0.9, metrics          # aprendeu o padrão, não só ruído
    assert metrics["mae"] < 1.0, metrics
    assert len(result.history["train_loss"]) == len(result.history["valid_loss"])
    assert result.best_epoch >= 1

    # 2. evaluate() calcula MAE/RMSE/R² corretamente numa previsão conhecida (modelo zerado -> prevê sempre 0)
    import torch
    toy = M.MLPRegressor(1, hidden_sizes=(2,))
    with torch.no_grad():
        for layer in (toy.net[0], toy.net[-1]):
            layer.weight.zero_()
            layer.bias.zero_()
    y_known = pd.Series([1.0, 2.0, 3.0, 4.0])
    m = M.evaluate(toy, pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0]}), y_known)
    assert np.isclose(m["mae"], y_known.abs().mean())
    assert np.isclose(m["rmse"], np.sqrt((y_known ** 2).mean()))

    # 3. save_model/load_model: previsões idênticas antes e depois de salvar em disco
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "modelo.pt"
    M.save_model(result.model, path)
    assert path.exists() and path.with_suffix(".json").exists()
    reloaded = M.load_model(path)
    pred_before = result.model(M._to_tensor(X_valid)).detach().numpy()
    pred_after = reloaded(M._to_tensor(X_valid)).detach().numpy()
    assert np.allclose(pred_before, pred_after, atol=1e-6)

    # 4. predict() usa o preprocessor (interface esperada pelo Integrante 3 / algoritmo genético)
    class FakePrep:
        feature_cols = list(X.columns)

        def transform(self, df, clip=None):
            return df.reindex(columns=self.feature_cols).astype(float)

    pred = M.predict(result.model, X_valid, FakePrep())
    assert np.allclose(pred, pred_before, atol=1e-6)

    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
