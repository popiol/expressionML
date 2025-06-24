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
    mode: PredictionMode = PredictionMode.EVALUATION
    model_with_noise: keras.Model | None = None

    @cached_property
    def model(self) -> keras.Model:
        self.raw_model.compile(optimizer="adam", loss="mse")
        return self.raw_model

    def predict(self, inputs: np.ndarray | list) -> np.ndarray:
        if self.mode == PredictionMode.TRAIN and self.model_with_noise is not None:
            return self.model_with_noise.predict(inputs, verbose=0)
        if self.mode == PredictionMode.EVALUATION:
            return self.model.predict(inputs, verbose=0)
        raise ValueError(f"Unknown mode: {self.mode}")

    def train(self, inputs: np.ndarray | list, outputs: np.ndarray):
        print(f"Training model with batch size {len(outputs)}")
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
    def combine_models(self, input_models: list[MlModel], inner_model: MlModel, output_model: MlModel) -> MlModel:
        inputs = [keras.layers.Input(shape=(input_model.model.input_shape[1],)) for input_model in input_models]
        combined_input = keras.layers.Concatenate()([model.model(x) for x, model in zip(inputs, input_models)])
        inner_output = inner_model.model(combined_input)
        output = output_model.model(inner_output)
        combined_model = keras.Model(inputs=inputs, outputs=output)
        version = ",".join(model.version for model in input_models + [inner_model, output_model])
        return MlModel(raw_model=combined_model, version=version)

    def get_model(self, version: str, input_size: int, output_size: int) -> MlModel:
        if version == "v1":
            model = self.v1(input_size, output_size)
        elif version == "v2":
            model = self.v2(input_size, output_size)
        elif version == "v3":
            model = self.v3(input_size, output_size)
        elif version == "lstm":
            model = self.lstm(input_size, output_size)
        else:
            raise ValueError(f"Unknown model version: {version}")
        return MlModel(raw_model=model, version=version)

    def v1(self, input_size: int, output_size: int) -> keras.Model:
        return keras.Sequential(
            [
                keras.layers.Input(shape=(input_size,)),
                keras.layers.Dense(100, activation="relu"),
                keras.layers.Dense(100, activation="relu"),
                keras.layers.Dense(output_size),
            ]
        )

    def v2(self, input_size: int, output_size: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(input_size,))
        l = inputs
        l = keras.layers.Dense(64, activation="relu")(l)
        for _ in range(4):
            l = keras.layers.Concatenate()([inputs, l])
            l = keras.layers.Dense(64, activation="relu")(l)
        l = keras.layers.Dense(output_size)(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v3(self, input_size: int, output_size: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(input_size,))
        l = inputs
        l1 = keras.layers.Dense(64, activation="relu")(l)
        l1 = keras.layers.Concatenate()([inputs, l])
        l1 = keras.layers.Dense(64, activation="relu")(l)
        l1 = keras.layers.Dense(output_size)(l)
        l2 = keras.layers.Dense(64, activation="relu")(l)
        l2 = keras.layers.Concatenate()([inputs, l])
        l2 = keras.layers.Dense(64, activation="relu")(l)
        l2 = keras.layers.Dense(output_size)(l)
        l3 = keras.layers.Dense(64, activation="relu")(l)
        l3 = keras.layers.Concatenate()([inputs, l])
        l3 = keras.layers.Dense(64, activation="relu")(l)
        l3 = keras.layers.Dense(output_size)(l)
        l4 = keras.layers.Dense(64, activation="relu")(l)
        l4 = keras.layers.Concatenate()([inputs, l])
        l4 = keras.layers.Dense(64, activation="relu")(l)
        l4 = keras.layers.Dense(output_size)(l)
        l5 = keras.layers.Dense(64, activation="relu")(l)
        l5 = keras.layers.Concatenate()([inputs, l])
        l5 = keras.layers.Dense(64, activation="relu")(l)
        l5 = keras.layers.Dense(4, activation="softmax")(l)
        l = keras.layers.Concatenate()([l1, l2, l3, l4])
        l = keras.layers.Reshape((4, output_size))(l)
        l = keras.layers.Dot(axes=1)([l, l5])
        return keras.Model(inputs=inputs, outputs=l)

    def lstm(self, input_size: int, output_size: int) -> keras.Model:
        return keras.Sequential(
            [
                keras.layers.Input(shape=(input_size,)),
                keras.layers.Reshape((input_size, 1)),
                keras.layers.LSTM(output_size, return_sequences=True),
                keras.layers.Flatten(),
                keras.layers.Dense(output_size),
            ]
        )

    def autoencoder(self, input_size: int, embedding_size: int) -> keras.Model:
        return keras.Sequential(
            [
                keras.layers.Input(shape=(input_size,)),
                keras.layers.Dense(embedding_size, activation="relu"),
                keras.layers.Dense(embedding_size, activation="relu"),
                keras.layers.Dense(embedding_size, activation="relu"),
                keras.layers.Dense(embedding_size, activation="relu"),
                keras.layers.Dense(input_size),
            ]
        )
