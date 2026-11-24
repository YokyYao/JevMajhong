import copy
from collections import defaultdict

from tools.generate_curriculum_kan_sources_v1 import plans, physical_wall
from tools.label_curriculum_kan_sources_v1 import observer_events


def test_teacher_history_hides_opponent_tiles_but_preserves_public_events():
    prefix = [dict(type='start_kyoku', tehais=[['1m']*13, ['2m']*13, ['3m']*13, ['4m']*13]),
              dict(type='tsumo', actor=0, pai='5mr'), dict(type='tsumo', actor=1, pai='6s'),
              dict(type='dahai', actor=0, pai='5mr', tsumogiri=True),
              dict(type='pon', actor=1, target=0, pai='5mr', consumed=['5m', '5m'])]
    original = copy.deepcopy(prefix)
    actual = observer_events(prefix, 1)
    assert actual[0]['tehais'] == [['?']*13, ['2m']*13, ['?']*13, ['?']*13]
    assert actual[1]['pai'] == '?'
    assert actual[2]['pai'] == '6s'
    assert actual[3:] == prefix[3:]
    assert prefix == original


def test_operation_and_physical_variants_stay_in_one_source_family():
    registered = plans(dict(seed_start=2026680000))
    templates = defaultdict(list)
    for plan in registered:
        templates[plan['motif'], plan['template']].append(plan)
    assert len(registered) == 192 and len(templates) == 32
    for group in templates.values():
        assert len(group) == 6
        assert len({p['group_id'] for p in group}) == 1
        assert len({p['split'] for p in group}) == 1
        assert {p['operation'] for p in group} == {'ankan', 'daiminkan', 'kakan'}
    assert not {p['group_id'] for p in registered if p['split'] == 'train'} & {
        p['group_id'] for p in registered if p['split'] == 'validation'}
    assert all(not 2026300000 <= p['seed'] < 2026300032 for p in registered)


def test_full_walls_are_physical_permutations_and_generation_is_repeatable():
    registered = plans(dict(seed_start=2026680000))
    assert registered == plans(dict(seed_start=2026680000))
    for plan in registered:
        wall, seat = physical_wall(plan)
        assert len(wall) == 136 and sorted(wall) == list(range(136))
        assert wall == physical_wall(plan)[0]
        assert seat == (0 if plan['operation'] == 'ankan' else 1)
