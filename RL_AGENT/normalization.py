import numpy as np


class RunningMeanStd:
    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)
        self.epsilon = float(epsilon)

    def update(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == len(self.mean.shape):
            arr = arr.reshape((1,) + self.mean.shape)
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = float(arr.shape[0])
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.epsilon)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    @classmethod
    def from_dict(cls, payload: dict, epsilon: float = 1e-4) -> "RunningMeanStd":
        mean = np.asarray(payload.get("mean", []), dtype=np.float64)
        var = np.asarray(payload.get("var", []), dtype=np.float64)
        count = float(payload.get("count", epsilon))
        obj = cls(shape=mean.shape, epsilon=epsilon)
        obj.mean = mean
        obj.var = var
        obj.count = count
        return obj
