"""Exercise 2: measure a complete synthetic CPU DataLoader iteration."""

from models import TrialReport


def run_trial(
    workers: int,
    prefetch_factor: int = 2,
    sample_count: int = 40,
    batch_size: int = 4,
    source_pause_ms: float = 5,
    consumer_pause_ms: float = 20,
    warmup_batches: int = 2,
) -> TrialReport:
    """Create isolated JSON records and preserve every ID and checksum."""
    raise NotImplementedError("Exercise 2: instrument next(), startup and CPU phases")
