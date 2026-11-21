"""Run the separately registered risk-recovery experiment, not original advancement."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.curriculum_parent import read,sha
from tools.curriculum_calls_recovery_parent import policy
from tools.attack_recovery_runtime import hooks
from tools import train_curriculum_attack as numerical


def run(registered):
    reg=read(registered)
    if (reg['curriculum']['run_id']!=policy()['experiment_run_id']
        or reg.get('original_calls_quality_passed') is not False
        or reg.get('experimental_transfer_not_qualified_advancement') is not True):
        raise ValueError('Explicit separately registered recovery experiment required')
    for path,expected in read(registered.parent.parent/'launch-code-binding-private.json').items():
        if sha(path)!=expected:raise ValueError('Prospectively bound recovery launcher/source changed: '+path)
    with hooks(train=numerical):numerical.run(registered)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    a=p.parse_args();run(a.registered.resolve())
