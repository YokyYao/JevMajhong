import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from jevany.model import PointerHead
from head_rl import fingerprint,write
from scripts.train_diagonal_value import BASE_FILES
from scripts.train_point_bias import prepare


class PointBiasPreparationTests(unittest.TestCase):
    def test_matching_audit_is_staged_but_changed_original_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'runs/source';source.mkdir(parents=True)
            original=root/'original';original.mkdir()
            checkpoint=source/'checkpoints/calibration-01';checkpoint.mkdir(parents=True)
            metadata={'head':PointerHead(4,3).state_dict(),'head_residual_dim':0,'temperature':1.}
            torch.save(metadata,original/'head.pt');torch.save(metadata,checkpoint/'head.pt')
            for name in BASE_FILES:
                (original/name).write_bytes(b'unchanged-runtime-input');shutil.copy2(original/name,checkpoint/name)
            original_sha=fingerprint(original/'head.pt');candidate_sha=fingerprint(checkpoint/'head.pt')
            cfg={'out':str(source),'reference':'original','development_seed_start':100,'development_seeds':64,
                 'confirmation_seed_start':200,'confirmation_seeds':128,'confirmation_min_ci95_lower':0,
                 'target_net_points_gain':0,'target_mean_net_points':0,'reward_scale':1000.,
                 'game_mode':'4p-red-east','reward_mode':'net_points','seed_scheme':'spaced-v1'}
            proto={'config':cfg,'sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                   'code_sha256':{},'reference_head_sha256':original_sha,
                   'reference_adapter_sha256':fingerprint(original/'adapter_model.safetensors')}
            write(source/'protocol.json',proto);(source/'private-regression.pt').write_bytes(b'audited-fit')
            dataset=root/'dataset';dataset.mkdir()
            files=[('rollout-stage-01.pt','rollout_sha256'),('training-games-stage-01.json','games_sha256'),
                   ('reference-controls-stage-01.json','controls_sha256')]
            provenance={'source':'dataset'}
            for name,key in files:
                (dataset/name).write_bytes(b'verified-source');provenance[key]=fingerprint(dataset/name)
            relative=str(checkpoint.relative_to(root))
            evidence={'audit_status':'passed','source_protocol_sha256':proto['sha256'],
                      'private_regression_sha256':fingerprint(source/'private-regression.pt'),
                      'data':{'sources':[provenance]},'candidate_heads':{relative:candidate_sha}}
            write(source/'fit-audit.json',evidence);write(source/'training-games-stage-cache-01.json',[{'seed':1}])
            (source/'native-development').mkdir();write(source/'native-development/protocol.json',{'code_sha256':{}})
            choice={'variant':1,'checkpoint':relative,'head_sha256':candidate_sha}
            with patch('scripts.train_point_bias.ROOT',root),patch('scripts.train_point_bias.ORIGINAL_HEAD',original_sha), \
                 patch('scripts.train_point_bias.audit_development',return_value={'selected':choice}):
                selected=prepare(source,root/'runs/final')
                self.assertEqual(selected['candidate_head_sha256'],candidate_sha)
                self.assertEqual(selected['config']['target_net_points_gain'],0)
                self.assertTrue((root/'runs/final/source-fit-audit.json').is_file())
                (original/'head.pt').write_bytes(b'changed-after-audit')
                with self.assertRaisesRegex(ValueError,'Original benchmark changed'):
                    prepare(source,root/'runs/another-final')


if __name__=='__main__':unittest.main()
