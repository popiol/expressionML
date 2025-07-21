from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
import numpy as np

from src.keras import keras, tf


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

    def clone(self):
        return MlModel(keras.models.clone_model(self.model), self.version)

    def add_noise(self):
        weights = self.model.get_weights()
        for layer in weights:
            layer += np.random.normal(loc=0.0, scale=0.1, size=layer.shape)
        self.model.set_weights(weights)


class FastReluRNN(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.output_dense = keras.layers.Dense(units, activation="relu")
        self.state_dense = keras.layers.Dense(units, activation="relu")

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        outputs = []
        state = None
        for i in range(inputs.shape[1]):
            x = inputs[:, i, :]
            if state is None:
                state = tf.zeros((batch_size, self.units), dtype=x.dtype)
            x_and_state = tf.concat([x, state], axis=-1)
            out = self.output_dense(x_and_state)
            outputs.append(tf.expand_dims(out, axis=1))
            state = self.state_dense(x_and_state)
        return tf.concat(outputs, axis=1)


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

    def get_model(self, version: str, in_objects: int, out_objects: int, n_features: int) -> MlModel:
        if version == "v0":
            model = self.v0(in_objects, out_objects, n_features)
        elif version == "v1":
            model = self.v1(in_objects, out_objects, n_features)
        elif version == "v2":
            model = self.v2(in_objects, out_objects, n_features)
        elif version == "v3":
            model = self.v3(in_objects, out_objects, n_features)
        elif version == "v4":
            model = self.v4(in_objects, out_objects, n_features)
        elif version == "v5":
            model = self.v5(in_objects, out_objects, n_features)
        elif version == "v6":
            model = self.v6(in_objects, out_objects, n_features)
        else:
            raise ValueError(f"Unknown model version: {version}")
        return MlModel(raw_model=model, version=version)

    def v0(self, in_objects: int, out_objects: int, n_features: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(in_objects, n_features))
        l = keras.layers.Flatten()(inputs)
        l = keras.layers.Dense(out_objects * n_features)(l)
        l = keras.layers.Reshape((out_objects, n_features))(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v1(self, in_objects: int, out_objects: int, n_features: int) -> keras.Model:
        # score: -0.192, 909s
        inputs = keras.layers.Input(shape=(in_objects, n_features))
        l = keras.layers.Flatten()(inputs)
        l = keras.layers.Dense(64, activation="relu")(l)
        l = keras.layers.Dense(64, activation="relu")(l)
        l = keras.layers.Dense(out_objects * n_features)(l)
        l = keras.layers.Reshape((out_objects, n_features))(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v2(self, in_objects: int, out_objects: int, n_features: int) -> keras.Model:
        # score:  -0.07788, 993s
        inputs = keras.layers.Input(shape=(in_objects, n_features))
        l = keras.layers.Flatten()(inputs)
        for _ in range(3):
            l = keras.layers.Dense(64, activation="relu")(l)
            l = keras.layers.Concatenate()([inputs, l])
            l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(out_objects * n_features)(l)
        l = keras.layers.Reshape((out_objects, n_features))(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v3(self, in_objects: int, out_objects: int, n_features: int) -> keras.Model:
        # score: -0.0147, 983s
        inputs = keras.layers.Input(shape=(in_objects, n_features))
        l = keras.layers.Flatten()(inputs)
        state = [l]
        for _ in range(3):
            l = keras.layers.Dense(64, activation="relu")(l)
            state.append(l)
            l = keras.layers.Concatenate()(state)
            l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(out_objects * n_features)(l)
        l = keras.layers.Reshape((out_objects, n_features))(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v4(self, in_objects: int, out_objects: int, n_features: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(in_objects, n_features))
        l = keras.layers.Lambda(lambda x: tf.reverse(x, axis=[-1]))(inputs)
        l = keras.layers.Permute((2, 1))(l)
        for size in [7, 4, out_objects]:
            l = FastReluRNN(size)(l)
        l = keras.layers.Permute((2, 1))(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v5(self, in_objects: int, out_objects: int, n_features: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(in_objects, n_features))
        l = keras.layers.Permute((2, 1))(inputs)
        n_convs = 10
        l = keras.layers.ZeroPadding1D(padding=n_convs)(l)
        for _ in range(n_convs-1):
            l = keras.layers.Conv1D(10, 3, activation="relu")(l)
        l = keras.layers.Conv1D(out_objects, 3, activation="relu")(l)
        l = keras.layers.Permute((2, 1))(l)
        return keras.Model(inputs=inputs, outputs=l)
