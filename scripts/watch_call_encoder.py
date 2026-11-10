"""Stdlib-only in-allocation encoder-training observer; no model changes."""
import argparse
import json
from pathlib import Path
import re
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--job',required=True)
    args=p.parse_args();cfg=json.loads(args.config.read_text());out=Path(cfg['out']);started=time.monotonic()
    while time.monotonic()-started<15000:
        if out.exists():
            path=out/'status.json';status=json.loads(path.read_text()) if path.exists() else {}
            log=Path('logs')/f'call-encoder-{args.job}.log';progress=[]
            if log.exists():
                with log.open('rb') as f:f.seek(max(0,log.stat().st_size-12000));tail=f.read().decode('utf-8','replace')
                progress=re.findall(r'(?:ep\d+ )?step (\d+)/(\d+)',tail)
            terminal=(out/'result.json').exists() or (out/'training-failure.json').exists()
            heartbeat={'job':args.job,'observed_unix':time.time(),'worker_status':status,
                'last_logged_optimizer_step':int(progress[-1][0]) if progress else 0,'terminal_record_exists':terminal}
            target=out/'observer-heartbeat.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(heartbeat,indent=2)+'\n');tmp.replace(target)
            if terminal:return
        time.sleep(60)


if __name__=='__main__':main()
