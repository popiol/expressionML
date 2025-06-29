import numpy as np

x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
signal = np.sin(x) + 0.5 * np.cos(2 * x)


def expand(x, w):
    return np.sin(w * x), np.cos(w * x)


def invert(y1, y2, w):
    return np.arctan2(y1, y2) / w


for x in signal:
    w = np.pi / (abs(x) + np.pi)
    y1, y2 = expand(x, w)
    print(y1, y2, w)
    x_recovered = invert(y1, y2, w)
    print(x, x_recovered)
