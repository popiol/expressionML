import time
from contextlib import contextmanager

@contextmanager
def timer(block_name: str = ""):
    start = time.time()
    yield
    end = time.time()
    print(f"{block_name} execution time: {end - start:.6f} seconds")
