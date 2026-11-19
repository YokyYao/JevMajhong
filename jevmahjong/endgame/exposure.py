"""Count actual admitted training record IDs before permitting model selection."""
from collections import defaultdict
import math


class TrainingExposure:
    def __init__(self, records, minimum_fractions=None):
        self.source_by_id = {}
        self.ids_by_source = defaultdict(set)
        self.seen = set()
        for record in records:
            rid = record['_meta']['id']
            if rid in self.source_by_id:
                raise ValueError('Duplicate training identity in exposure ledger')
            source = record['questions']['action']['src']
            self.source_by_id[rid] = source
            self.ids_by_source[source].add(rid)
        self.minimum_fractions = dict(minimum_fractions or {})
        for source, fraction in self.minimum_fractions.items():
            if (source not in self.ids_by_source or isinstance(fraction, bool)
                    or not isinstance(fraction, (float, int))
                    or not math.isfinite(fraction) or not 0 < fraction <= 1):
                raise ValueError('Unknown or invalid registered exposure requirement')

    def observe(self, ids):
        ids = set(ids)
        if not ids <= set(self.source_by_id):
            raise ValueError('Unregistered record entered actual training batch')
        self.seen.update(ids)

    def snapshot(self):
        rows = {}
        ready = True
        for source, ids in self.ids_by_source.items():
            seen = len(ids & self.seen)
            required = math.ceil(len(ids) * self.minimum_fractions.get(source, 0))
            rows[source] = {'unique_seen': seen, 'total': len(ids), 'minimum_required': required}
            ready = ready and seen >= required
        return {'unique_records_seen': len(self.seen), 'total_records': len(self.source_by_id),
                'by_source': rows, 'minimum_exposure_met': ready}


def exposure_early_stop_update(original, value, best, stale, *, step, exposure_met,
                               other_guards_passed, **kwargs):
    # Keep step0 as a valid no-change choice. Before sufficient coverage, an
    # otherwise safe diagnostic is neither a candidate nor a patience failure.
    # A real replay/placement guard failure still consumes normal patience.
    if step > 0 and not exposure_met and other_guards_passed:
        if best is None:
            raise ValueError('Exposure-gated selection requires baseline evaluation')
        return best, 0, False
    return original(value, best, stale, **kwargs)
