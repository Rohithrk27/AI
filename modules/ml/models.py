"""
ML Models Module
Defines 7 surrogate models with a unified interface for training, prediction, and uncertainty estimation.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")


class SurrogateModel:
    """Base class for all surrogate models."""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.params = {}

    def train(self, X, y, params=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_with_uncertainty(self, X):
        """Return (mean, std). Default: std = 0."""
        mean = self.predict(X)
        std = np.zeros_like(mean)
        return mean, std

    def get_default_params(self):
        return {}

    def get_name(self):
        return self.__class__.__name__


# ──────────────────────────────────────────────
# 1. Gaussian Process Regression
# ──────────────────────────────────────────────
class GPModel(SurrogateModel):
    """Gaussian Process Regression using scikit-learn."""

    def get_default_params(self):
        return {"kernel": "rbf", "alpha": 1e-6, "n_restarts": 10}

    def train(self, X, y, params=None):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel, RationalQuadratic

        p = {**self.get_default_params(), **(params or {})}
        self.params = p

        # Build kernel
        kernel_name = p.get("kernel", "rbf")
        alpha = p.get("alpha", 1e-6)
        n_restarts = p.get("n_restarts", 10)

        if kernel_name == "matern32":
            kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel()
        elif kernel_name == "matern52":
            kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
        elif kernel_name == "rational_quadratic":
            kernel = ConstantKernel() * RationalQuadratic() + WhiteKernel()
        else:  # rbf
            kernel = ConstantKernel() * RBF() + WhiteKernel()

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts,
            random_state=42,
        )
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        mean, std = self.model.predict(X, return_std=True)
        return mean, std

    def get_name(self):
        return "Gaussian Process"


# ──────────────────────────────────────────────
# 2. Random Forest
# ──────────────────────────────────────────────
class RandomForestModel(SurrogateModel):
    """Random Forest Regression."""

    def get_default_params(self):
        return {"n_estimators": 200, "max_depth": None, "min_samples_split": 2,
                "min_samples_leaf": 1, "max_features": 1.0}

    def train(self, X, y, params=None):
        from sklearn.ensemble import RandomForestRegressor

        p = {**self.get_default_params(), **(params or {})}
        self.params = p

        self.model = RandomForestRegressor(
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=p.get("max_depth"),
            min_samples_split=int(p.get("min_samples_split", 2)),
            min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            max_features=p.get("max_features", 1.0),
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        """Uncertainty from variance across trees."""
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return mean, std

    def get_name(self):
        return "Random Forest"


# ──────────────────────────────────────────────
# 3. XGBoost
# ──────────────────────────────────────────────
class XGBoostModel(SurrogateModel):
    """XGBoost Regression."""

    def get_default_params(self):
        return {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_alpha": 0.0, "reg_lambda": 1.0}

    def train(self, X, y, params=None):
        from xgboost import XGBRegressor

        p = {**self.get_default_params(), **(params or {})}
        self.params = p

        self.model = XGBRegressor(
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=int(p.get("max_depth", 6)),
            learning_rate=p.get("learning_rate", 0.1),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_alpha=p.get("reg_alpha", 0.0),
            reg_lambda=p.get("reg_lambda", 1.0),
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        """Approximate uncertainty via quantile predictions."""
        mean = self.predict(X)
        # Use a simple heuristic: predict with subsets of trees
        n_trees = self.model.n_estimators
        step = max(1, n_trees // 10)
        preds = []
        for n in range(step, n_trees + 1, step):
            p = self.model.predict(X, iteration_range=(0, n))
            preds.append(p)
        std = np.std(preds, axis=0) if len(preds) > 1 else np.zeros_like(mean)
        return mean, std

    def get_name(self):
        return "XGBoost"


# ──────────────────────────────────────────────
# 4. LightGBM
# ──────────────────────────────────────────────
class LightGBMModel(SurrogateModel):
    """LightGBM Regression."""

    def get_default_params(self):
        return {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.1,
                "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8}

    def train(self, X, y, params=None):
        from lightgbm import LGBMRegressor

        p = {**self.get_default_params(), **(params or {})}
        self.params = p

        self.model = LGBMRegressor(
            n_estimators=int(p.get("n_estimators", 200)),
            num_leaves=int(p.get("num_leaves", 31)),
            learning_rate=p.get("learning_rate", 0.1),
            min_child_samples=int(p.get("min_child_samples", 20)),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=42,
            verbosity=-1,
        )
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        mean = self.predict(X)
        # Approximate via staged predictions
        n_trees = self.model.n_estimators
        step = max(1, n_trees // 10)
        preds = []
        for n in range(step, n_trees + 1, step):
            p = self.model.predict(X, num_iteration=n)
            preds.append(p)
        std = np.std(preds, axis=0) if len(preds) > 1 else np.zeros_like(mean)
        return mean, std

    def get_name(self):
        return "LightGBM"


# ──────────────────────────────────────────────
# 5. CatBoost
# ──────────────────────────────────────────────
class CatBoostModel(SurrogateModel):
    """CatBoost Regression."""

    def get_default_params(self):
        return {"iterations": 200, "depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 3.0}

    def train(self, X, y, params=None):
        from catboost import CatBoostRegressor

        p = {**self.get_default_params(), **(params or {})}
        self.params = p

        self.model = CatBoostRegressor(
            iterations=int(p.get("iterations", 200)),
            depth=int(p.get("depth", 6)),
            learning_rate=p.get("learning_rate", 0.1),
            l2_leaf_reg=p.get("l2_leaf_reg", 3.0),
            random_state=42,
            verbose=0,
        )
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        mean = self.predict(X)
        # CatBoost supports virtual ensembles for uncertainty
        try:
            preds = self.model.virtual_ensembles_predict(X, prediction_type="TotalUncertainty")
            std = preds[:, 1] if preds.shape[1] > 1 else np.zeros_like(mean)
        except Exception:
            std = np.zeros_like(mean)
        return mean, std

    def get_name(self):
        return "CatBoost"


# ──────────────────────────────────────────────
# 6. Artificial Neural Network
# ──────────────────────────────────────────────
class ANNModel(SurrogateModel):
    """Neural Network Regression using PyTorch."""

    def get_default_params(self):
        return {"hidden_layers": 2, "hidden_units": 64, "learning_rate": 0.001,
                "dropout": 0.1, "epochs": 200, "batch_size": 32}

    def train(self, X, y, params=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        p = {**self.get_default_params(), **(params or {})}
        self.params = p

        n_features = X.shape[1]
        hidden_units = int(p.get("hidden_units", 64))
        n_layers = int(p.get("hidden_layers", 2))
        dropout = p.get("dropout", 0.1)
        lr = p.get("learning_rate", 0.001)
        epochs = int(p.get("epochs", 200))
        batch_size = int(p.get("batch_size", 32))

        # Build network
        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self.device = torch.device("cpu")
        self.net.to(self.device)

        # Training
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(X)), shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.net.train()
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = self.net(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        self.net.eval()
        self.model = self.net
        self.is_trained = True

    def predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            pred = self.net(X_tensor).cpu().numpy().ravel()
        return pred

    def predict_with_uncertainty(self, X):
        """MC Dropout for uncertainty estimation."""
        import torch
        self.net.train()  # Enable dropout
        preds = []
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            for _ in range(30):  # 30 forward passes
                p = self.net(X_tensor).cpu().numpy().ravel()
                preds.append(p)
        self.net.eval()
        preds = np.array(preds)
        return preds.mean(axis=0), preds.std(axis=0)

    def get_name(self):
        return "Neural Network"


# ──────────────────────────────────────────────
# 7. Support Vector Regression
# ──────────────────────────────────────────────
class SVRModel(SurrogateModel):
    """Support Vector Regression."""

    def get_default_params(self):
        return {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"}

    def train(self, X, y, params=None):
        from sklearn.svm import SVR

        p = {**self.get_default_params(), **(params or {})}
        self.params = p

        self.model = SVR(
            kernel=p.get("kernel", "rbf"),
            C=p.get("C", 10.0),
            epsilon=p.get("epsilon", 0.1),
            gamma=p.get("gamma", "scale"),
        )
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        """Distance-based uncertainty heuristic for SVR."""
        mean = self.predict(X)
        # Use distance to support vectors as uncertainty proxy
        try:
            decision = self.model.decision_function(X)
            std = 1.0 / (np.abs(decision) + 1e-8)
            # Normalize to reasonable range
            std = std / std.max() * mean.std() if mean.std() > 0 else std
        except Exception:
            std = np.ones_like(mean) * 0.1
        return mean, std

    def get_name(self):
        return "SVR"


# ──────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────
def get_all_models() -> dict:
    """Return a dictionary of all available surrogate models."""
    return {
        "Gaussian Process": GPModel(),
        "Random Forest": RandomForestModel(),
        "XGBoost": XGBoostModel(),
        "LightGBM": LightGBMModel(),
        "CatBoost": CatBoostModel(),
        "Neural Network": ANNModel(),
        "SVR": SVRModel(),
    }
