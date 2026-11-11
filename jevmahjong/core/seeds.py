"""Separate per-episode wall streams in RiichiEnv 0.4.11.

Upstream shuffles hand h with splitmix64(episode_seed + h). Adjacent raw
episode seeds therefore share shifted hand streams. Reserve 2**32 hand
seeds per logical episode; the simulator aborts long before that budget.
"""
SEED_STRIDE = 1 << 32
MAX_ENVIRONMENT_STEPS = 100000


def engine_seed(episode_id, scheme="spaced-v1"):
    if scheme == "raw":
        if not 0 <= episode_id < (1 << 64):
            raise ValueError("Raw seed must be uint64")
        return episode_id
    if scheme != "spaced-v1" or not 0 <= episode_id < (1 << 32):
        raise ValueError("spaced-v1 requires a uint32 logical episode ID")
    return episode_id * SEED_STRIDE
