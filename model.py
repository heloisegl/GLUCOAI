"""Rede neural (MLP em PyTorch) para estimar a glicose média (mg/dL) a partir das features
de comportamento/bio geradas por preprocessing.py.

Uso típico (ver notebooks/03_rede_neural.ipynb):
    import pandas as pd, preprocessing as P, model as M

    X_train = pd.read_csv("data/processed/model_ready/day_level/X_train.csv")
    y_train = pd.read_csv("data/processed/model_ready/day_level/y_train.csv")["target_gl_mean"]
    X_valid = pd.read_csv("data/processed/model_ready/day_level/X_valid.csv")
    y_valid = pd.read_csv("data/processed/model_ready/day_level/y_valid.csv")["target_gl_mean"]

    result = M.train_model(X_train, y_train, X_valid, y_valid)
    M.evaluate(result.model, X_valid, y_valid)
    M.save_model(result.model, "models/day_level_mlp.pt")

    prep = P.load_preprocessor("data/processed/model_ready/day_level/preprocessor.pkl")
    M.predict(result.model, df_novo, prep)   # df_novo = linhas com as colunas originais (não escaladas)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)
        self.n_features = n_features
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainResult:
    model: MLPRegressor
    history: dict          # {"train_loss": [...], "valid_loss": [...]} (MSE por época)
    best_epoch: int


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy(dtype="float32")
    return torch.as_tensor(np.array(x, dtype="float32"))


def train_model(X_train, y_train, X_valid, y_valid, hidden_sizes: tuple[int, ...] = (64, 32),
                 dropout: float = 0.2, lr: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs: int = 500, patience: int = 30, seed: int = 42, verbose: bool = True) -> TrainResult:
    """Treina a MLP com early stopping na perda (MSE) de validação; devolve o modelo da melhor época."""
    torch.manual_seed(seed)
    Xtr, ytr = _to_tensor(X_train), _to_tensor(y_train)
    Xva, yva = _to_tensor(X_valid), _to_tensor(y_valid)

    model = MLPRegressor(Xtr.shape[1], hidden_sizes, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history: dict = {"train_loss": [], "valid_loss": []}
    best_state, best_valid, best_epoch, wait = None, float("inf"), 0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            valid_loss = loss_fn(model(Xva), yva).item()
        history["train_loss"].append(loss.item())
        history["valid_loss"].append(valid_loss)

        if valid_loss < best_valid - 1e-6:
            best_valid, best_epoch, wait = valid_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping na época {epoch} (melhor: {best_epoch}, valid_mse={best_valid:.3f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, history, best_epoch)


def evaluate(model: MLPRegressor, X, y) -> dict:
    """MAE, RMSE e R² (y na unidade original, mg/dL)."""
    model.eval()
    with torch.no_grad():
        pred = model(_to_tensor(X)).numpy()
    y = np.asarray(y, dtype="float64")
    err = pred - y
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def predict(model: MLPRegressor, df_novo: pd.DataFrame, prep, clip: float | None = None) -> np.ndarray:
    """Aplica o MESMO preprocessor do treino (`prep.transform`, de preprocessing.load_preprocessor)
    e roda a rede. `df_novo` tem as colunas originais, não escaladas (ex.: um plano do algoritmo genético)."""
    X = prep.transform(df_novo, clip=clip)
    model.eval()
    with torch.no_grad():
        return model(_to_tensor(X)).numpy()


def save_model(model: MLPRegressor, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    config = {"n_features": model.n_features, "hidden_sizes": list(model.hidden_sizes), "dropout": model.dropout}
    path.with_suffix(".json").write_text(json.dumps(config, indent=2))


def load_model(path) -> MLPRegressor:
    path = Path(path)
    config = json.loads(path.with_suffix(".json").read_text())
    model = MLPRegressor(config["n_features"], tuple(config["hidden_sizes"]), config["dropout"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
