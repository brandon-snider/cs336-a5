from vllm import SamplingParams

import time
import random


def ordered_filename(prefix: str):
    suffix = f"{str(int(time.time() * 1000))[-10:]}_{random.randint(0, 9999):04d}"
    return f"{prefix}_{suffix}"


default_sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)
