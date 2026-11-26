"""Versioned public win-source input for future endgame training.

Never infer a ron payer from the previous discard when the trigger was kan.
The caller must pass the current native public event, not synthetic replay
initialization history. This module does not score hands or produce labels.
"""
import copy

SCHEMA = 'public-win-source-v1'


def public_tile(tile):
    """Normalize native MJAI honors/red fives to the model's MPSZ format."""
    honors = dict(zip(('E', 'S', 'W', 'N', 'P', 'F', 'C'), ('1z','2z','3z','4z','5z','6z','7z')))
    if not isinstance(tile, str):
        raise ValueError('Invalid public tile')
    tile = honors.get(tile, tile)
    if tile in ('5mr', '5pr', '5sr'):
        tile = '0'+tile[1]
    if (len(tile) != 2 or tile[1] not in 'mpsz'
        or tile[0] not in ('1234567' if tile[1] == 'z' else '0123456789')):
        raise ValueError('Invalid public tile')
    return tile


def project_trigger(event, observer):
    """Allowlist one current public event; opponent draw tiles stay hidden."""
    if type(observer) is not int or observer not in range(4):
        raise ValueError('Invalid observing seat')
    kind, actor = event.get('type'), event.get('actor')
    if kind not in ('dahai', 'tsumo', 'kakan', 'ankan'):
        raise ValueError('No supported current win trigger')
    if type(actor) is not int or actor not in range(4):
        raise ValueError('Invalid public trigger actor')
    if kind == 'tsumo' and actor != observer:
        return dict(type=kind, actor=actor)
    tile = event.get('pai')
    if kind == 'ankan':
        consumed = event.get('consumed')
        if not isinstance(consumed, list) or len(consumed) != 4:
            raise ValueError('Ankan trigger requires its public four-tile identity')
        normal = [public_tile(t).replace('0', '5') for t in consumed]
        if len(set(normal)) != 1:
            raise ValueError('Ankan tiles are inconsistent')
        tile = normal[0]
    return dict(type=kind, actor=actor, tile=public_tile(tile))


def with_win_context(request, wins, trigger):
    """Annotate exact hora keys using verified public trigger and legal kind."""
    result = copy.deepcopy(request)
    state = result['state']
    seat = state['seat']
    criteria = result['questions']['action']['criteria']
    if type(seat) is not int or seat not in range(4) or not wins:
        raise ValueError('A legal win window and valid own seat are required')
    if set(wins) != {k for k, c in criteria.items() if c.get('type') == 'hora'}:
        raise ValueError('Native win identities differ from the exact action menu')
    context = {}
    for key, native in wins.items():
        kind, tile = native['kind'], native['tile']
        actor = trigger.get('actor')
        if type(actor) is not int or actor not in range(4):
            raise ValueError('Invalid win-source seat')
        if tile != trigger.get('tile'):
            raise ValueError('Legal winning tile differs from current public trigger')
        if kind == 'tsumo':
            if trigger.get('type') != 'tsumo' or actor != seat:
                raise ValueError('Tsumo must follow own public draw')
            source = 'self_draw'
        elif kind == 'ron':
            if actor == seat:
                raise ValueError('Ron cannot come from self')
            if trigger.get('type') == 'dahai':
                if not state['rivers'][actor].split() or state['rivers'][actor].split()[-1] != tile:
                    raise ValueError('Discard trigger differs from source river')
                source = 'discard'
            elif trigger.get('type') in ('kakan', 'ankan'):
                source = 'robbed_'+trigger['type']
            else:
                raise ValueError('Ron requires a public discard or kan trigger')
        else:
            raise ValueError('Unsupported native win kind')
        context[key] = dict(kind=kind, source=source, source_seat=actor, tile=tile)
    if 'win_context' in state:
        raise ValueError('Never silently overwrite another input contract')
    state['win_context'] = dict(schema=SCHEMA, candidates=context)
    return result


def build_win_request(obs, current_public_event):
    """Native adapter; no environment, hidden hand or wall is accepted."""
    from riichienv import ActionType
    from jevmahjong.core.state import build_request, tile_name
    request, actions, features = build_request(obs)
    keys = list(request['questions']['action']['criteria'])
    wins = {key: dict(kind='ron' if action.action_type == ActionType.RON else 'tsumo',
        tile=tile_name(action.tile)) for key, action in zip(keys, actions)
        if action.action_type in (ActionType.RON, ActionType.TSUMO)}
    trigger = project_trigger(current_public_event, obs.player_id)
    return with_win_context(request, wins, trigger), actions, features
