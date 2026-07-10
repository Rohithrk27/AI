import numpy as np
import warnings

class SurrogateModel:
    """Base class for all surrogate models."""
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.params = {}

    def get_default_params(self):
        return {}

    def train(self, X, y, params=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_with_uncertainty(self, X):
        """Return (mean, std). Default: std = 0."""
        mean = self.predict(X)
        std = np.zeros_like(mean)
        return mean, std

    def get_name(self):
        return "BaseModel"

class GPModel(SurrogateModel):
    """Gaussian Process Regression."""

    def get_default_params(self):
        return {"kernel": "Matern", "alpha": 1e-10, "n_restarts_optimizer": 5}

    def train(self, X, y, params=None):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic
        from sklearn.exceptions import ConvergenceWarning

        p = {**self.get_default_params(), **(params or {})}
        self.params = p

        if p.get("kernel") == "RBF":
            kernel = RBF(length_scale=1.0)
        elif p.get("kernel") == "RationalQuadratic":
            kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
        else:
            kernel = Matern(length_scale=1.0, nu=1.5)

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(p.get("alpha", 1e-10)),
            n_restarts_optimizer=int(p.get("n_restarts_optimizer", 5)),
            random_state=42,
            normalize_y=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        mean, std = self.model.predict(X, return_std=True)
        return mean, std

    def get_name(self):
        return "Gaussian Process"

def get_all_models() -> dict:
    """Return a dictionary of all available surrogate models."""
    return {
        "Gaussian Process": GPModel(),
    }
