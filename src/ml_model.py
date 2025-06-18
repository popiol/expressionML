from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property

import numpy as np

from src.keras import keras


class PredictionMode(Enum):
    TRAIN = auto()
    EVALUATION = auto()


@dataclass
class MlModel:
    raw_model: keras.Model
    version: str
    batch_size: int
    mode: PredictionMode = PredictionMode.EVALUATION
    model_with_noise: keras.Model | None = None
    batch: tuple[np.ndarray, np.ndarray] | None = None

    @cached_property
    def model(self) -> keras.Model:
        self.raw_model.compile(optimizer="adam", loss="mse")
        return self.raw_model

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        input_batch = inputs.reshape((1, -1))
        if self.mode == PredictionMode.TRAIN and self.model_with_noise is not None:
            return self.model_with_noise.predict(input_batch, verbose=0)[0]
        if self.mode == PredictionMode.EVALUATION:
            return self.model.predict(input_batch, verbose=0)[0]
        raise ValueError(f"Unknown mode: {self.mode}")

    def train(self, inputs: np.ndarray, outputs: np.ndarray, n_epochs: int = 1):
        input_batch = inputs.reshape((1, -1))
        output_batch = outputs.reshape((1, -1))
        if self.batch is None or len(self.batch[0]) >= self.batch_size - 1:
            self.batch = (input_batch, output_batch)
        else:
            self.batch = (np.concatenate((self.batch[0], input_batch)), np.concatenate((self.batch[1], output_batch)))
            input_batch = self.batch[0]
            output_batch = self.batch[1]
        if len(input_batch) >= self.batch_size - 1:
            print("Training model with full batch")
            self.model.fit(input_batch, output_batch, verbose=0, epochs=self.batch_size * n_epochs)

    def train_batch(self, inputs: np.ndarray, outputs: np.ndarray):
        print(f"Training model with batch size {len(inputs)}")
        self.model.fit(inputs, outputs, verbose=0, epochs=len(inputs))

    def set_training_mode(self):
        self.mode = PredictionMode.TRAIN
        self.model_with_noise = keras.models.clone_model(self.model)
        weights = self.model.get_weights()
        for layer in weights:
            layer += np.random.normal(loc=0.0, scale=0.1, size=layer.shape)
        assert self.model_with_noise is not None
        self.model_with_noise.set_weights(weights)

    def set_evaluation_mode(self):
        self.mode = PredictionMode.EVALUATION
        self.model_with_noise = None


@dataclass
class MlModelFactory:
    batch_size: int

    def combine_models(self, input_models: list[MlModel], inner_model: MlModel, output_model: MlModel) -> MlModel:
        inputs = [keras.layers.Input(shape=(input_model.model.input_shape[1],)) for input_model in input_models]
        combined_input = keras.layers.Concatenate()(inputs)
        inner_output = inner_model.model(combined_input)
        output = output_model.model(inner_output)
        combined_model = keras.Model(inputs=inputs, outputs=output)
        version = ",".join(model.version for model in input_models + [inner_model, output_model])
        return MlModel(raw_model=combined_model, version=version, batch_size=self.batch_size)

    def get_model(self, version: str, input_size: int, output_size: int) -> MlModel:
        if version == "v1":
            model = self.v1(input_size, output_size)
        elif version == "lstm":
            model = self.lstm(input_size, output_size)
        else:
            raise ValueError(f"Unknown model version: {version}")
        return MlModel(raw_model=model, version=version, batch_size=self.batch_size)

    def v1(self, input_size: int, output_size: int) -> keras.Model:
        return keras.Sequential(
            [
                keras.layers.Input(shape=(input_size,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(output_size),
            ]
        )

    def lstm(self, input_size: int, output_size: int) -> keras.Model:
        return keras.Sequential(
            [
                keras.layers.Input(shape=(input_size,)),
                keras.layers.Dense(output_size * 64),
                keras.layers.Reshape((output_size, 64)),
                keras.layers.LSTM(output_size),
                keras.layers.Dense(output_size),
            ]
        )
