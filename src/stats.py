from functools import cached_property

import numpy as np


class Stats:
    class StatsHolder:
        def __init__(self, value_shape: tuple):
            self.mean = np.zeros(value_shape)
            self.mean_squared = np.zeros(value_shape)
            self.std = np.zeros(value_shape)
            self.min = np.full(value_shape, np.nan)
            self.max = np.full(value_shape, np.nan)
            self.count = 0
            self.samples: list[list[float]] = []

    @staticmethod
    def from_batch(values: list | np.ndarray):
        stats = Stats()
        stats.add_batch(values)
        return stats

    def __init__(self):
        self.reset()

    def reset(self):
        try:
            del self.value_shape
        except AttributeError:
            pass
        try:
            del self._stats
        except AttributeError:
            pass

    @cached_property
    def value_shape(self) -> tuple[int, ...]:
        raise ValueError("Stats not initialized")

    @cached_property
    def _stats(self):
        return self.StatsHolder(self.value_shape)

    def __bool__(self):
        try:
            self._stats
            return True
        except ValueError:
            return False

    def add_single_value(self, value: float | list | np.ndarray):
        self.add_batch([value])

    def add_batch(self, values: list | np.ndarray):
        batch_size = len(values)
        self.value_shape = np.shape(values[0])
        self._stats.mean = (
            self._stats.mean * self._stats.count + np.mean(values, axis=0) * batch_size
        ) / (self._stats.count + batch_size)
        self._stats.mean_squared = (
            self._stats.mean_squared * self._stats.count
            + np.power(values, 2).mean(axis=0) * batch_size
        ) / (self._stats.count + batch_size)
        self._stats.std = np.power(self._stats.mean_squared - np.power(self._stats.mean, 2), 0.5)
        self._stats.min = np.nanmin([self._stats.min, np.nanmin(values, axis=0)], axis=0)
        self._stats.max = np.nanmax([self._stats.max, np.nanmax(values, axis=0)], axis=0)
        self._stats.count += batch_size
        self.add_sample(values)

    def add_sample(self, values: list | np.ndarray):
        if len(self._stats.samples) > 1:
            return
        values = np.array(values)
        for sample in values:
            sample = np.array(sample)
            while np.ndim(sample) > 1:
                sample = np.take(sample, 0, axis=-1)
            if np.ndim(sample) == 0:
                sample = [sample]
            sample = np.array(sample)
            if np.all(sample == 0):
                continue
            if len(self._stats.samples) == 1:
                if len(sample) == 1:
                    if sample * self._stats.samples[0] >= 0:
                        continue
                elif (sample[-1] - sample[0]) * (
                    self._stats.samples[0][-1] - self._stats.samples[0][0]
                ) >= 0:
                    continue
            self._stats.samples.append(sample.tolist())
            break

    @staticmethod
    def from_dict(data: dict):
        stats = Stats()
        stats.value_shape = np.shape(data["mean"])
        stats._stats = stats.StatsHolder(stats.value_shape)
        stats._stats.mean = np.array(data["mean"])
        stats._stats.std = np.array(data["std"])
        stats._stats.min = np.array(data["min"])
        stats._stats.max = np.array(data["max"])
        stats._stats.count = data["count"]
        stats._stats.samples = data["samples"]
        return stats

    @property
    def mean(self):
        return self._stats.mean

    @property
    def std(self):
        return self._stats.std

    @property
    def min(self):
        return self._stats.min

    @property
    def max(self):
        return self._stats.max

    @property
    def count(self):
        return self._stats.count

    @property
    def samples(self):
        return self._stats.samples

    def to_dict(self):
        return {
            "mean": self._stats.mean.tolist(),
            "std": self._stats.std.tolist(),
            "min": self._stats.min.tolist(),
            "max": self._stats.max.tolist(),
            "count": self._stats.count,
            "samples": self._stats.samples,
        }

    def to_dict_per_feature(self, feature_names: list[str]) -> dict[str, dict[str, float]]:
        assert len(feature_names) == self.value_shape[-1]
        result = {
            name: {
                "mean": self._stats.mean[..., index],
                "std": self._stats.std[..., index],
                "min": self._stats.min[..., index],
                "max": self._stats.max[..., index],
            }
            for index, name in enumerate(feature_names)
        }
        result[feature_names[0]]["samples"] = self._stats.samples
        return result

    def to_list_per_feature(self) -> list[dict[str, float]]:
        return [
            {
                "mean": self._stats.mean[..., index].tolist(),
                "std": self._stats.std[..., index].tolist(),
                "min": self._stats.min[..., index].tolist(),
                "max": self._stats.max[..., index].tolist(),
            }
            for index in range(self.value_shape[-1])
        ]
