"""Data coverage and selection rules for a separate, immutable training stage."""
from feature_training_guard import FeatureSkillGuard


def remaining_indices(order, size, *, epoch, next_microbatch, records_seen,
                      batch, world_size, p_none_pair):
    """Recover unseen source requests only for the registered single-rank recipe."""
    if epoch != 0 or batch != 1 or world_size != 1 or p_none_pair != 0:
        raise ValueError('Unseen recovery requires first epoch, batch1, one rank, no paired augmentation')
    if len(order) != size or sorted(order) != list(range(size)):
        raise ValueError('Saved order must be a full source permutation')
    if not 0 <= next_microbatch <= size or next_microbatch != records_seen:
        raise ValueError('Training cursor and actual exposure differ')
    return order[next_microbatch:]


class UpdatedFeatureSkillGuard(FeatureSkillGuard):
    """Rank updated policies by NLL without retesting unchanged parent weights."""
    def update(self, original, value, best, stale, maximize=True, min_delta=0.):
        if self.step == 0:
            return best, stale, False
        return super().update(original, value, best, stale,
                              maximize=maximize, min_delta=min_delta)


def verify_preparation(root, config, prepared):
    from call_policy import digest
    if not prepared['complete'] or prepared['config'] != config or prepared['model_test_opened']:
        raise ValueError('Incomplete, changed, or test-exposed continuation preparation')
    for name, expected in prepared['frozen_files'].items():
        if digest(root / name) != expected:
            raise ValueError('Continuation source changed: ' + name)
    for split in ('train', 'validation'):
        if digest(root / config['data'] / (split + '.jsonl')) != prepared['files'][split]:
            raise ValueError('Continuation data changed: ' + split)
