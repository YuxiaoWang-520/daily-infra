"""Exercise 2: implement the CPU replay-equivalence experiment."""


def run_trial(
    checkpointed: bool,
    preserve_rng_state: bool = True,
    dropout_p: float = 0.4,
    blocks: int = 4,
    width: int = 8,
    batch_size: int = 2,
    seed: int = 2026,
):
    """Return outputs, gradients, SGD updates, RNG sample and saved-storage metadata.

    Read code/README.md for the required result fields and measurement scope.
    """
    raise NotImplementedError("Exercise 2: compare identical seeded CPU training steps")
