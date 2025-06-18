import numpy as np
import pytest

from src.coder import AdvancedCoder


@pytest.fixture
def coder():
    return AdvancedCoder(32)


def test_encode_decode_integer(coder):
    value = 123
    vector = coder.encode_integer(value)
    embedding = coder.to_embedding(vector)
    decoded = coder.decode_integer(embedding)
    assert decoded == value


def test_encode_decode_float(coder):
    value = -3.14
    vector = coder.encode_float(value)
    embedding = coder.to_embedding(vector)
    decoded = coder.decode_float(embedding)
    assert np.isclose(decoded, value)


def test_encode_decode_text(coder):
    text = "hello world"
    vector = coder.encode_text(text)
    embedding = coder.to_embedding(vector)
    decoded = coder.decode_text(embedding)
    assert text == decoded
