from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from src.keras import keras


@dataclass
class MlModel:
    raw_model: keras.Model
    version: str

    @cached_property
    def model(self) -> keras.Model:
        self.raw_model.compile(optimizer="adam", loss="mse")
        return self.raw_model

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.model.predict(inputs, verbose=0)

    def train(self, inputs: np.ndarray, outputs: np.ndarray):
        print(f"Training model with batch size {len(outputs)}")
        self.model.fit(inputs, outputs, verbose=0, epochs=len(outputs))

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

    def clone(self):
        return MlModel(keras.models.clone_model(self.model), self.version)

    def add_noise(self):
        weights = self.model.get_weights()
        for layer in weights:
            layer += np.random.normal(loc=0.0, scale=0.1, size=layer.shape)
        self.model.set_weights(weights)


@dataclass
class MlModelFactory:
    def get_model_with_noise(self, model: MlModel) -> MlModel:
        new_model = model.clone()
        new_model.add_noise()
        return new_model

    def combine_models(
        self, input_models: list[MlModel], inner_model: MlModel, output_model: MlModel
    ) -> MlModel:
        inputs = [
            keras.layers.Input(shape=(input_model.model.input_shape[1],))
            for input_model in input_models
        ]
        combined_input = keras.layers.Concatenate()(
            [model.model(x) for x, model in zip(inputs, input_models)]
        )
        inner_output = inner_model.model(combined_input)
        output = output_model.model(inner_output)
        combined_model = keras.Model(inputs=inputs, outputs=output)
        version = ",".join(model.version for model in input_models + [inner_model, output_model])
        return MlModel(raw_model=combined_model, version=version)

    def get_model(self, version: str, input_size: int, output_size: int) -> MlModel:
        if version == "v0":
            model = self.v0(input_size, output_size)
        elif version == "v1":
            model = self.v1(input_size, output_size)
        elif version == "v2":
            model = self.v2(input_size, output_size)
        elif version == "v3":
            model = self.v3(input_size, output_size)
        elif version == "v4":
            model = self.v4(input_size, output_size)
        elif version == "v5":
            model = self.v5(input_size, output_size)
        elif version == "v6":
            model = self.v6(input_size, output_size)
        else:
            raise ValueError(f"Unknown model version: {version}")
        return MlModel(raw_model=model, version=version)

    def v0(self, input_size: int, output_size: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(input_size,))
        l = inputs
        l = keras.layers.Dense(output_size)(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v1(self, input_size: int, output_size: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(input_size,))
        l = inputs
        l = keras.layers.Dense(64, activation="relu")(l)
        l = keras.layers.Dense(64, activation="relu")(l)
        l = keras.layers.Dense(output_size)(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v2(self, input_size: int, output_size: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(input_size,))
        l = inputs
        for _ in range(2):
            l = keras.layers.Dense(64, activation="relu")(l)
            l = keras.layers.Concatenate()([inputs, l])
            l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(output_size)(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v3(self, input_size: int, output_size: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(input_size,))
        l = inputs
        state = [l]
        for _ in range(3):
            l = keras.layers.Dense(64, activation="relu")(l)
            state.append(l)
            l = keras.layers.Concatenate()(state)
            l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(output_size)(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v4(self, input_size: int, output_size: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(input_size,))
        l = inputs
        l = keras.layers.Reshape((3, input_size // 3))(l)[:, :2]
        l1 = keras.layers.Dense(4)(l)
        l2 = keras.layers.Dense(1)(l1)
        l3 = keras.layers.Flatten()(l2)
        l3 = keras.layers.Dense(1)(l3)
        l4 = keras.layers.Dense(input_size // 3 - 4)(l)
        l4 = keras.layers.Flatten()(l4)
        l = keras.layers.Concatenate()([l3, l4])
        l5 = keras.layers.Dense(64, activation="relu")(l)
        l5 = keras.layers.Concatenate()([l3, l5])
        l5 = keras.layers.Dense(64, activation="relu")(l5)
        l5 = keras.layers.Dense(64, activation="relu")(l5)
        l5 = keras.layers.Dense(64, activation="relu")(l5)
        l6 = keras.layers.Flatten()(l1)
        l5 = keras.layers.Concatenate()([l6, l5])
        l = keras.layers.Dense(output_size)(l5)
        return keras.Model(inputs=inputs, outputs=l)

    def v5(self, input_size: int, output_size: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(input_size,))
        l = inputs
        for _ in range(2):
            l = keras.layers.Dense(64, activation="softmax")(l)
            l = keras.layers.Dense(64)(l)
        l = keras.layers.Dense(output_size)(l)
        return keras.Model(inputs=inputs, outputs=l)
