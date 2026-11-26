"""Resolve an explicitly declared historical Python source, never data aliases.

This preserves original manifest keys and records both historical and current
identities. It does not run historical code, prove source-family exclusion, or
admit a model to the final comparison.
"""
import hashlib
from pathlib import Path
import re

METHOD = 'exact_protocol_bound_historical_python_source_alias_v1'


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def resolve(original, expected, declaration, *, root, original_manifest,
            manifest_sha256, protocol_code_bindings):
    """Require an exact explicit alias to the same protocol-bound .py module.

    The caller supplies code bindings from independently verified actual
    ancestral protocols. This function checks their membership and the current
    files; it does not establish that caller's complete model ancestry.
    """
    root = Path(root).resolve()
    original = Path(original)
    manifest = Path(original_manifest)
    if not original.is_absolute() or not manifest.is_absolute():
        raise ValueError('Canonical absolute original and manifest paths required')
    if original.resolve() != original or manifest.resolve() != manifest:
        raise ValueError('No source or manifest symlink/path redirection allowed')
    try:
        relative = original.relative_to(root / 'jevmahjong')
    except ValueError as error:
        raise ValueError('Only original project Python source may have an alias') from error
    if original.suffix != '.py' or not original.is_file():
        raise ValueError('Existing Python source required; no data/weight aliases')
    if re.fullmatch('[0-9a-f]{64}', expected or '') is None:
        raise ValueError('Exact historical SHA256 required')
    if set(declaration) != {'method', 'original_path', 'expected_sha256',
            'original_manifest', 'manifest_sha256', 'archived_path'}:
        raise ValueError('Exact explicit source-alias schema required')
    if (declaration['method'] != METHOD or declaration['original_path'] != str(original)
            or declaration['expected_sha256'] != expected
            or declaration['original_manifest'] != str(manifest)
            or declaration['manifest_sha256'] != manifest_sha256
            or digest(manifest) != manifest_sha256):
        raise ValueError('Alias must be bound to the unchanged original manifest declaration')
    import json
    inputs = json.loads(manifest.read_text())
    if inputs.get(str(original)) != expected:
        raise ValueError('Historical identity must actually occur in the original manifest')
    current = digest(original)
    if current == expected:
        raise ValueError('A matching current source needs no historical alias')
    archive = Path(declaration['archived_path'])
    if not archive.is_absolute() or archive.resolve() != archive:
        raise ValueError('Canonical absolute archived path without symlink required')
    try:
        parts = archive.relative_to(root / 'runs' / 'code').parts
    except ValueError as error:
        raise ValueError('Only an explicit archived project code package is eligible') from error
    if (len(parts) < 3 or re.fullmatch('[0-9a-f]{64}', parts[0]) is None
            or Path(*parts[1:]) != Path('jevmahjong') / relative):
        raise ValueError('Archive must have the identical package-relative Python module path')
    if protocol_code_bindings.get(str(archive)) != expected:
        raise ValueError('Archive must be an exact code input of a verified actual ancestor protocol')
    if not archive.is_file() or digest(archive) != expected:
        raise ValueError('Archived source bytes must match the original historical identity')
    return dict(method=METHOD, original_path=str(original), expected_sha256=expected,
        observed_current_sha256=current, original_manifest=str(manifest),
        manifest_sha256=manifest_sha256, archived_path=str(archive),
        archived_sha256=expected, role='historical_python_source_only',
        actual_protocol_code_membership_checked=True, original_key_preserved=True,
        original_source_or_manifest_rewritten=False, source_code_executed=False,
        source_family_exclusion_proved=False, final_confirmation_admitted=False)
