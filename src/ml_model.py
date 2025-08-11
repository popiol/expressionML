from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from src.keras import keras, tf


@dataclass
class MlModel:
    raw_model: keras.Model
    version: str

    @cached_property
    def model(self) -> keras.Model:
        self.raw_model.compile(optimizer="adam", loss="huber")
        return self.raw_model

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.model.predict(inputs, verbose=0)

    def train(self, inputs: np.ndarray, outputs: np.ndarray, epochs_multiplier: int = 1):
        print(f"Training model with batch size {len(outputs)}")
        self.model.fit(
            inputs,
            outputs,
            verbose=0,
            epochs=len(outputs) * epochs_multiplier,
            callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=3)],
        )

    def clone(self):
        return MlModel(keras.models.clone_model(self.model), self.version)

    def add_noise(self):
        weights = self.model.get_weights()
        for layer in weights:
            layer += np.random.normal(loc=0.0, scale=0.1, size=layer.shape)
        self.model.set_weights(weights)


class TrainableMask(keras.layers.Layer):
    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.mask = self.add_weight(
            shape=shape, initializer="ones", trainable=True, name="trainable_mask"
        )

    def call(self, inputs):
        return inputs * self.mask


class PadLastDimLayer(keras.layers.Layer):
    def __init__(self, lpad, rpad, **kwargs):
        super().__init__(**kwargs)
        self.lpad = lpad
        self.rpad = rpad

    def call(self, inputs):
        paddings = [[0, 0] for _ in range(len(inputs.shape))]
        paddings[-1][0] = self.lpad
        paddings[-1][1] = self.rpad
        return tf.pad(inputs, paddings)


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

    def get_model(
        self, version: str, in_objects: int, in_features: int, out_objects: int, out_features: int
    ) -> MlModel:
        if version == "v0":
            model = self.v0(in_objects, in_features, out_objects, out_features)
        elif version == "v1":
            model = self.v1(in_objects, in_features, out_objects, out_features)
        elif version == "v2":
            model = self.v2(in_objects, in_features, out_objects, out_features)
        elif version == "v3":
            model = self.v3(in_objects, in_features, out_objects, out_features)
        elif version == "v4":
            model = self.v4(in_objects, in_features, out_objects, out_features)
        elif version == "v5":
            model = self.v5(in_objects, in_features, out_objects, out_features)
        elif version == "v6":
            model = self.v6(in_objects, in_features, out_objects, out_features)
        else:
            raise ValueError(f"Unknown model version: {version}")
        return MlModel(raw_model=model, version=version)

    def v0(
        self, in_objects: int, in_features: int, out_objects: int, out_features: int
    ) -> keras.Model:
        inputs = keras.layers.Input(shape=(in_objects, in_features))
        l = keras.layers.Flatten()(inputs)
        l = keras.layers.Dense(out_objects * out_features)(l)
        l = keras.layers.Reshape((out_objects, out_features))(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v1(
        self, in_objects: int, in_features: int, out_objects: int, out_features: int
    ) -> keras.Model:
        inputs = keras.layers.Input(shape=(in_objects, in_features))
        l = keras.layers.Flatten()(inputs)
        for index in range(5):
            size = 2 ** (10 - index)
            l = keras.layers.Dense(size, activation="relu")(l)
        l = keras.layers.Dense(out_objects * out_features)(l)
        l = keras.layers.Reshape((out_objects, out_features))(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v2(
        self, in_objects: int, in_features: int, out_objects: int, out_features: int
    ) -> keras.Model:
        inputs = keras.layers.Input(shape=(in_objects, in_features))
        l = keras.layers.Flatten()(inputs)
        l1 = l
        for _ in range(3):
            l = keras.layers.Dense(64, activation="relu")(l)
            l = keras.layers.Concatenate()([l1, l])
            l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(out_objects * out_features)(l)
        l = keras.layers.Reshape((out_objects, out_features))(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v3(
        self, in_objects: int, in_features: int, out_objects: int, out_features: int
    ) -> keras.Model:
        inputs = keras.layers.Input(shape=(in_objects, in_features))
        l = keras.layers.Flatten()(inputs)
        state = [l]
        for _ in range(3):
            l = keras.layers.Dense(64, activation="relu")(l)
            state.append(l)
            l = keras.layers.Concatenate()(state)
            l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(out_objects * out_features)(l)
        l = keras.layers.Reshape((out_objects, out_features))(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v4(
        self, in_objects: int, in_features: int, out_objects: int, out_features: int
    ) -> keras.Model:
        inputs = keras.layers.Input(shape=(in_objects, in_features))
        l = keras.layers.Permute((2, 1))(inputs)
        n_convs = 10
        l = keras.layers.ZeroPadding1D(padding=n_convs)(l)
        for _ in range(n_convs - 1):
            l = keras.layers.Conv1D(10, 3, activation="relu")(l)
        l = keras.layers.Conv1D(out_objects, 3, activation="relu")(l)
        l = keras.layers.Permute((2, 1))(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v5(
        self, in_objects: int, in_features: int, out_objects: int, out_features: int
    ) -> keras.Model:
        inputs = keras.layers.Input(shape=(in_objects, in_features))
        l = inputs
        l = keras.layers.Flatten()(l)
        iterations = 20
        state = [l]
        max_state_size = 5
        base = 9
        offset = out_features // base
        for index in range(iterations):
            l = PadLastDimLayer(-l.shape[-1] % (offset * base), 0)(l)
            l = keras.layers.Reshape((-1, offset, base))(l)
            l = keras.layers.Permute((2, 1, 3))(l)
            l = keras.layers.Reshape((offset, -1))(l)
            l = keras.layers.Dense(base, activation="relu")(l)
            l = keras.layers.Reshape((offset, -1, base))(l)
            l = keras.layers.Permute((2, 1, 3))(l)
            l = keras.layers.Flatten()(l)
            state.append(l)
            if len(state) > max_state_size:
                state = state[-max_state_size:]
            l = keras.layers.Concatenate()(state)
        l = PadLastDimLayer(-l.shape[-1] % out_features, 0)(l)
        l = keras.layers.Reshape((-1, out_features))(l)
        l = keras.layers.Permute((2, 1))(l)
        l = keras.layers.Dense(out_objects)(l)
        l = keras.layers.Permute((2, 1))(l)
        return keras.Model(inputs=inputs, outputs=l)

    def v6(
        self, in_objects: int, in_features: int, out_objects: int, out_features: int
    ) -> keras.Model:
        inputs = keras.layers.Input(shape=(in_objects, in_features))
        l = keras.layers.Permute((2, 1))(inputs)
        n_convs = 20
        state = [l]
        max_state_size = 5
        base = 3
        for _ in range(n_convs - 1):
            l = keras.layers.ZeroPadding1D(padding=1)(l)
            l = keras.layers.Conv1D(base, 3, activation="relu")(l)
            state.append(l)
            if len(state) > max_state_size:
                state = state[-max_state_size:]
            l = keras.layers.Concatenate()(state)
        l = keras.layers.ZeroPadding1D(padding=1)(l)
        l = keras.layers.Conv1D(out_objects, 3, activation="relu")(l)
        l = keras.layers.Permute((2, 1))(l)
        return keras.Model(inputs=inputs, outputs=l)
