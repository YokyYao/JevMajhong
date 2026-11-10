"""Bounded Slurm observer: persistent health/events, without pretending to wake chat."""
import argparse
import fcntl
import getpass
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, data):
    """Atomic observer state, without loading a training runtime on the observer."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def observation_complete(cfg, state):
    if cfg.get('evaluation_stage') == 'feature_cache':
        return (state.get('training') or {}).get('status') == 'feature_cache_complete'
    if state.get('skipped'): return True
    if (state.get('strength') or {}).get('status') != 'development_complete': return False
    if cfg.get('evaluation_stage') != 'independent_confirmation': return True
    audit, qualification = state.get('audit') or {}, state.get('qualification') or {}
    return audit.get('status') == 'passed_complete' and audit.get('complete') is True and 'qualified' in qualification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--train-job', type=int)
    parser.add_argument('--eval-job', type=int, required=True)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--max-seconds', type=int, default=21600)
    args = parser.parse_args()
    if args.interval < 10 or args.interval > 60: raise ValueError('Monitor interval must be10..60seconds')
    cfg = json.loads(args.config.read_text()); run = ROOT / cfg['out']
    monitor = run.with_name(run.name + '-monitor'); monitor.mkdir(exist_ok=True)
    lock = (monitor / 'observer.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def terminated(signum, frame):
        write(monitor / 'action-required.json', {'reason': 'observer_terminated', 'signal': signum,
            'next_action': 'Inspect worker status and logs; do not restart blindly', 'agent_notification_sent': False})
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminated)
    previous, empty_polls, started = None, 0, time.monotonic()
    with (monitor / 'events-private.jsonl').open('a') as events:
        while time.monotonic() - started < args.max_seconds:
            now = datetime.now(timezone.utc).isoformat()
            jobs = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%R'],
                                  capture_output=True, text=True, timeout=20)
            wanted = {str(job) for job in (args.train_job, args.eval_job) if job is not None}
            queue = '\n'.join(line for line in jobs.stdout.splitlines() if line.split('|')[0] in wanted)
            state = {'training': read(run / 'status.json'), 'strength': read(run / 'strength-status.json'),
                'skipped': read(run / 'development-skipped.json'), 'queue': queue,
                'audit': read(run / 'development-audit.json'), 'qualification': read(run / 'qualification.json'),
                'queue_query_error': jobs.stderr.strip() if jobs.returncode else None,
                'train_job': args.train_job, 'eval_job': args.eval_job}
            if jobs.returncode == 0 and not state['queue']: empty_polls += 1
            else: empty_polls = 0
            final = observation_complete(cfg, state)
            reason = 'completed' if final else None
            if 'DependencyNeverSatisfied' in state['queue'] or (empty_polls >= 2 and not final): reason = 'worker_failed_or_missing_result'
            progress = [p.stat().st_mtime for p in [run / 'status.json', run / 'strength-status.json'] if p.exists()]
            state['progress_stalled_over_20min'] = bool(progress and '|RUNNING|' in queue and time.time() - max(progress) > 1200)
            write(monitor / 'heartbeat.json', {'observed_at_utc': now, 'status': reason or 'watching', **state,
                'can_wake_codex_thread': False, 'agent_continuations_depend_on_active_goal': True})
            encoded = json.dumps(state, sort_keys=True)
            if encoded != previous or reason:
                events.write(json.dumps({'observed_at_utc': now, 'event': reason or 'state_changed', **state}) + '\n'); events.flush()
                print(json.dumps({'observed_at_utc': now, 'event': reason or 'state_changed', 'queue': state['queue']}), flush=True)
                previous = encoded
            if reason:
                summaries = {s: read(run / 'native-development' / f'{s}-summary.json') for s in ('closed', 'mixed')}
                action = 'Review complete development and choose next training/independent confirmation; never auto-deploy' if final else 'Inspect worker logs and repair; no blind rerun'
                if final and cfg.get('evaluation_stage') == 'feature_cache':
                    action = 'Review frozen feature runtime audit; only train/validation are available for subsequent work; test stays sealed'
                elif final and cfg.get('evaluation_stage') == 'independent_confirmation':
                    action = 'Review complete independent qualification; if weak diagnose and use fresh training data; desktop/human regressions still required'
                write(monitor / 'action-required.json', {'reason': reason, 'next_action': action,
                    'summaries': summaries, 'agent_notification_sent': False, 'observed_at_utc': now})
                return
            time.sleep(args.interval)
    write(monitor / 'action-required.json', {'reason': 'observer_time_limit', 'agent_notification_sent': False})


if __name__ == '__main__': main()
