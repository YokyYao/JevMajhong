"""Correct meld-presence versus actual menzen/open aggregation, append-only.

The original successful native audit remains immutable. Its action availability
is unchanged; its scope_open_hand/scope_closed_hand labels actually counted
nonempty/empty meld records and must not be used as actual openness counts.
"""
import argparse
from collections import Counter
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write


def structure(melds):
    if not isinstance(melds, list): raise ValueError('Verified public meld list required')
    kinds = []
    for meld in melds:
        if not isinstance(meld, dict): raise ValueError('Public meld objects required')
        kind = meld.get('type')
        if type(kind) not in (int, str) or kind not in (0, 1, 2, 3, 4, 'chi', 'pon', 'daiminkan', 'ankan', 'kakan'):
            raise ValueError('Known native public meld type required')
        kinds.append(kind)
    opened = any(kind not in (3, 'ankan') for kind in kinds)
    ankan = any(kind in (3, 'ankan') for kind in kinds)
    return dict(open=opened, closed=not opened, any_meld=bool(kinds), ankan=ankan,
                closed_with_ankan=ankan and not opened)


def audit(out):
    gpu_only(); out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'runs/curriculum-defensive-scope-v1/audit-v1'; original = read(source/'summary-private.json')
    if original['status'] != 'prospective_defensive_scope_native_source_availability_audited_not_fit_admitted':
        raise ValueError('Actual complete original native interface audit required')
    reg = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-plan/registered.json')
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from riichienv import MeldType
    from jevmahjong.data.base_lora_pool import phase
    from tools.audit_curriculum_riichi_risk_scope_v1 import COHORTS
    from tools.curriculum_riichi_risk_scope_policy_v1 import in_scope
    if int(MeldType.Ankan) != 3 or {int(getattr(MeldType, n)) for n in ('Chi', 'Pon', 'Daiminkan', 'Ankan', 'Kakan')} != set(range(5)):
        raise ValueError('Pinned native meld-type mapping differs')
    files = dict(read(source/'binding-private.json')['files'])
    paths = [Path(__file__), ROOT/'tests/core/test_curriculum_defensive_openness_v1.py',
        ROOT/'docs/experiments/plans/curriculum-defensive-openness-20261123.md',
        source/'summary-private.json', source/'binding-private.json']
    files.update({str(p): sha(p) for p in paths})
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Original native source binding changed')
    totals = {}; checked = 0; reclassified = 0
    for cohort, (directory, required) in COHORTS.items():
        manifest = read(ROOT/'runs'/directory/'audit-v1/summary-private.json')
        if manifest['status'] != required: raise ValueError('Immutable cohort audit required')
        for split, spec in manifest['files'].items():
            path = ROOT/spec['path']
            if sha(path) != spec['sha256']: raise ValueError('Native-verified public source changed')
            counted = Counter()
            for line in path.open():
                row = json.loads(line); state = row['state']; q = row['questions']['action']
                types = [c['type'] for c in q['criteria'].values()]
                detail = structure(state['melds'][state['seat']]); scope = in_scope(types, state)
                counted['rows'] += 1; counted['scope'] += scope
                counted['all_open'] += detail['open']; counted['all_closed'] += detail['closed']
                counted['all_closed_with_ankan'] += detail['closed_with_ankan']
                if scope:
                    counted['scope_open'] += detail['open']; counted['scope_closed'] += detail['closed']
                    counted['scope_closed_with_ankan'] += detail['closed_with_ankan']
                    counted['scope_any_meld'] += detail['any_meld']; counted['scope_no_meld'] += not detail['any_meld']
                    expected_phase = 'open_discard' if detail['open'] else 'closed_discard'
                    if phase(row) != expected_phase or row['_meta']['phase'] != expected_phase:
                        raise ValueError('Verified source/trainer phase does not match actual menzen status')
            previous = original['counts'][cohort+':'+split]
            if (counted['rows'] != spec['records'] or counted['scope'] != previous.get('scope', 0)
                    or counted['scope_any_meld'] != previous.get('scope_open_hand', 0)
                    or counted['scope_no_meld'] != previous.get('scope_closed_hand', 0)
                    or counted['scope_open']+counted['scope_closed'] != counted['scope']):
                raise ValueError('Original meld-presence counter or complete source coverage not reproduced')
            totals[cohort+':'+split] = dict(counted)
            checked += counted['rows']; reclassified += counted['scope_closed_with_ankan']
    if checked != 45351 or any(sha(p) != h for p, h in files.items()):
        raise ValueError('Complete unchanged original inputs required')
    result = dict(status='defensive_scope_actual_openness_append_only_correction_audited', checked_rows=checked,
        counts=totals, closed_with_ankan_scope_rows_previously_named_open=reclassified,
        invalid_original_field_names=['scope_open_hand', 'scope_closed_hand'],
        original_fields_mean_meld_present_or_absent_not_actual_openness=True,
        original_guard_eligibility_scope_and_target_availability_unchanged=True,
        original_source_meta_phase_matches_actual_openness_in_scope=True,
        prior45351_native_input_parity_inherited_not_rerun=True, new_native_replays=0,
        original_summary_and_bound_code_unchanged=True, current_calls_fit_unchanged=True,
        student_optimizer_updates=0, risk_fit_or_route_admitted=False, raw_private=True)
    write(out/'summary-private.json', result); write(out/'binding-private.json', dict(files=files))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    audit(p.parse_args().out.resolve())
