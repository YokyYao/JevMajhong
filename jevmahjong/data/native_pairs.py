"""Load compatible, complete native collections without repeating whole walls."""
import json
from pathlib import Path
import torch
from jevmahjong.core.io import digest
from jevmahjong.training.collect import validate
from jevmahjong.training.utility import relabel, validate_config

ENCODER_FILES = ('adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'chat_template.jinja')


def select_decisions(partitions, mode='all', scopes=None):
    if mode not in ('all', 'kan_windows', 'open_tenpai_riichi'): raise ValueError('Unknown decision filter')
    if mode == 'all' and scopes is None: return partitions
    from jevmahjong.policy.kan import is_kan_request
    from jevmahjong.policy.defense import is_defense_request
    accepted = is_defense_request if mode == 'open_tenpai_riichi' else is_kan_request
    return {split:[r for r in rows if (mode == 'all' or accepted(r['public_request']))
        and (scopes is None or r['scope'] in scopes)] for split,rows in partitions.items()}


def encoder_identity(protocol):
    files = protocol.get('model_files')
    if files is None:
        files = {'discard': protocol['reference_files'], 'call': protocol['teacher_files']}
    normalized = {role: {Path(p).name: value for p, value in files[role].items()}
        for role in ('discard', 'call')}
    return {role: {name: normalized[role][name] for name in ENCODER_FILES} for role in normalized}


def load_sources(roots, model_files=None, reward=None, *, allow_unlabeled=False):
    if reward is not None: validate_config(reward)
    if isinstance(roots, (str, Path)):
        roots = [roots]
    roots = [Path(p).resolve() for p in roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError('Unique, nonempty collection paths required')
    partitions = {'train': [], 'held': []}; hashes = {}; sources = []; seen_walls = set()
    identity = encoder_identity({'model_files': model_files}) if model_files else None
    conditions = None
    for root in roots:
        marker = root / 'collection-result.json'; protocol_file = root / 'protocol.json'
        collection = json.loads(marker.read_text()); protocol = json.loads(protocol_file.read_text())
        config = protocol['config']
        from .native_states import INPUT_METHODS, validate as validate_inputs
        unlabeled = protocol.get('collection_method') in INPUT_METHODS
        if unlabeled and not allow_unlabeled:
            raise ValueError('Native decision inputs have no reward labels; collect paired futures before fitting')
        behavior = protocol.get('behavior_heads')
        if behavior and digest(Path(behavior['checkpoint'])) != behavior['sha256']:
            raise ValueError('Source behavior head checkpoint changed')
        if collection['status'] != 'collection_complete':
            raise ValueError('Complete native collection required')
        signature = tuple(config[k] for k in ('game_mode', 'seed_scheme', 'fast_gamma'))
        if config['seed_scheme'] != 'spaced-v1' or (conditions is not None and signature != conditions):
            raise ValueError('Sources need compatible rules, reward and independent wall streams')
        conditions = signature
        current = encoder_identity(protocol)
        if identity is not None and current != identity:
            raise ValueError('Source frozen encoders differ')
        identity = current
        if protocol.get('collection_method') == 'paired_future_ev_v1':
            for path, checksum in protocol['source_files'].items():
                if (path in hashes and hashes[path] != checksum) or digest(Path(path)) != checksum:
                    raise ValueError('Future-data original input changed')
                hashes[path] = checksum
        expected = set(range(config['training_seed_start'],
            config['training_seed_start'] + config['training_seeds_per_stage']))
        if seen_walls & expected:
            raise ValueError('Whole wall repeated across sources')
        seen_walls.update(expected); actual = set(); counts = {'train': 0, 'held': 0}; rows = 0
        for path in sorted((root / 'walls-private').glob('*.pt')):
            checksum = digest(path)
            if checksum != json.loads(path.with_suffix('.audit.json').read_text())['sha256']:
                raise ValueError('Source shard changed')
            payload = torch.load(path, map_location='cpu', weights_only=False); seed = payload['seed']
            if seed not in expected or seed in actual:
                raise ValueError('Foreign or duplicate whole-wall shard')
            actual.add(seed)
            split = 'held' if (seed - config['training_seed_start']) % 4 == 3 else 'train'
            if protocol.get('collection_method') == 'paired_future_ev_v1':
                from .future_pairs import validate_rows
                selected = validate_rows(payload,seed,config,protocol,reward)
            elif unlabeled:
                selected = validate_inputs(payload,seed,config)
            else:
                validate(payload, seed, config)
                selected = relabel(payload, config, reward) if reward else payload['rows']
            counts[split] += 1; partitions[split].extend(selected); rows += len(selected)
            hashes[str(path)] = checksum
        if actual != expected or len(actual) != collection['walls'] or rows != collection['rows']:
            raise ValueError('Source collection coverage differs')
        sources.append({'path': str(root), 'protocol_sha256': digest(protocol_file),
            'collection_sha256': digest(marker), 'walls': len(actual), 'rows': rows,
            'train_walls': counts['train'], 'held_walls': counts['held'],
            'seed_start': config['training_seed_start'], 'seeds': len(expected),
            **({'behavior_heads': behavior} if behavior else {})})
    return partitions, hashes, sources
