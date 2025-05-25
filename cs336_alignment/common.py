import time
import random


def ordered_filename(prefix: str):
    suffix = f"{str(int(time.time() * 1000))[-10:]}_{random.randint(0, 9999):04d}"
    return f"{prefix}_{suffix}"
