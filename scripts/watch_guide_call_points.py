"""Stdlib-only in-allocation heartbeat; never changes policy or starts jobs."""
import argparse
import fcntl
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--job', required=True); p.add_argument('--max-seconds', type=int, default=15000)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); out = Path(cfg['out'])
    started = time.monotonic(); previous = None; changed = started; lock = None
    while time.monotonic() - started < args.max_seconds:
        if out.exists():
            if lock is None:
                lock = (out / 'observer.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = out / 'status.json'; status = json.loads(path.read_text()) if path.exists() else {}
            serialized = json.dumps(status, sort_keys=True)
            if serialized != previous: changed = time.monotonic(); previous = serialized
            terminal = (out / 'fit-result.json').exists()
            heartbeat = {'job':args.job,'observed_unix':time.time(),'worker_status':status,
                'terminal_fit_record_exists':terminal,'seconds_without_progress':time.monotonic()-changed,
                'needs_attention':not terminal and time.monotonic()-changed > 1200,
                'authority':'observe only; active Codex goal handles failures and next experiments'}
            target = out / 'observer-heartbeat.json'; tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(heartbeat, indent=2)+'\n'); tmp.replace(target)
            print(json.dumps(heartbeat), flush=True)
            if terminal: return
        time.sleep(60)


if __name__ == '__main__': main()
