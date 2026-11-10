"""In-allocation training observer; training failure and terminal files are authoritative."""
import argparse
import json
from pathlib import Path
import re
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--job',required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());out=Path(cfg['out']);start=time.monotonic()
    while time.monotonic()-start<15000:
        status=out/'status.json';log=Path('logs')/f'clean-policy-{a.job}.log'
        progress=[]
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-16000));tail=f.read().decode('utf8','replace')
            progress=re.findall(r'(?:ep\d+ )?step (\d+)/(\d+)',tail)
        terminal=(out/'result.json').exists() or (out/'training-failure.json').exists()
        value={'job':a.job,'observed_unix':time.time(),'worker_status':json.loads(status.read_text()) if status.exists() else {},
            'last_logged_optimizer_step':int(progress[-1][0]) if progress else 0,'terminal_record_exists':terminal}
        target=out/'observer-heartbeat.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(target)
        if terminal:return
        time.sleep(60)


if __name__=='__main__':main()
