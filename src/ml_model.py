from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from src.keras import keras


class PredictionMode(Enum):
    TRAIN = auto()
    EVALUATION = auto()


@dataclass
class MlModel:
    model: keras.Model
    version: str
    mode: PredictionMode = PredictionMode.EVALUATION
    model_with_noise: keras.Model | None = None

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self.mode == PredictionMode.TRAIN and self.model_with_noise is not None:
            return self.model_with_noise.predict(inputs)
        if self.mode == PredictionMode.EVALUATION:
            return self.model.predict(inputs)
        raise ValueError(f"Unknown mode: {self.mode}")

    def train(self, inputs: np.ndarray, outputs: np.ndarray):
        self.model.fit(inputs, outputs)

    def set_training_mode(self):
        self.mode = PredictionMode.TRAIN
        self.model_with_noise = keras.models.clone_model(self.model)
        weights = self.model.get_weights()
        for layer in weights:
            layer += np.random.normal(loc=0.0, scale=0.4, size=layer.shape)
        assert self.model_with_noise is not None
        self.model_with_noise.set_weights(weights)

    def set_evaluation_mode(self):
        self.mode = PredictionMode.EVALUATION
        self.model_with_noise = None


class MlModelFactory:
    def get_model(self, version: str, input_size: int, output_size: int) -> MlModel:
        if version == "v1":
            model = self.v1(input_size, output_size)
        else:
            raise ValueError(f"Unknown model version: {version}")
        model.compile(optimizer="adam", loss="mse")
        return MlModel(model=model, version=version)

    def v1(self, input_size: int, output_size: int) -> keras.Model:
        return keras.Sequential(
            [
                keras.layers.Input(shape=(input_size,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(output_size, activation="linear"),
            ]
        )
