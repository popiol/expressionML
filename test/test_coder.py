import numpy as np
import pytest

from src.lookup_coder import LookupCoder

coders = [LookupCoder(i) for i in [64]]


@pytest.mark.parametrize("coder", coders)
def test_encode_decode_integer(coder):
    for value in [0, 1, 123, 785, -1, -9, -785]:
        embedding = coder.encode(value)
        decoded = coder.decode(embedding, int)
        print(value, embedding, decoded)
        assert decoded == value


@pytest.mark.parametrize("coder", coders)
def test_encode_decode_float(coder):
    for value in [0.0, -3.14, -0.001, 0.001, 3.14]:
        embedding = coder.encode(value)
        decoded = coder.decode(embedding, float)
        print(value, embedding, decoded)
        assert np.isclose(decoded, value)


@pytest.mark.parametrize("coder", coders)
def test_encode_decode_text(coder):
    text = "hello world"
    embedding = coder.encode(text)
    decoded = coder.decode(embedding, str)
    assert text == decoded
