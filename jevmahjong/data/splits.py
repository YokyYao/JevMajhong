import hashlib

def split_for_match(match_id):
    # All seats, rounds and decisions of a match must remain in one split.
    bucket = int.from_bytes(hashlib.sha256(match_id.encode()).digest()[:8], "big") % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"
