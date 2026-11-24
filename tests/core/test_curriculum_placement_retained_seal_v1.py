"""A new placement seal cannot bypass retained-parent exits or missing pools."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import seal_curriculum_placement_retained_parent_v1 as seal


def phases():
    queue = {role+'_job': str(index) for index, role in enumerate(('prepare', 'fit', 'audit', 'capture'), 101)}
    cap = {'scheduler_proofs': {}}; receipts = {}
    for role in ('prepare', 'fit', 'audit', 'capture'):
        job = queue[role+'_job']
        raw = (f'JobId={job} JobState=COMPLETED ExitCode=0:0 NodeList=ws-fixture '
            'StartTime=2026-11-24T00:00:00 EndTime=2026-11-24T01:00:00 '
            f'AllocTRES=cpu=4,gres/gpu=1 Command={seal.retained.ENTRY.resolve()}')
        receipt = dict(job_id=job, phase=role, exit_code=0)
        path = seal.ROOT/'logs'/f'curriculum-yaku-parent-exit-{job}.json'; receipts[path] = receipt
        if role == 'capture': captured = dict(raw=raw, exit=receipt); capture_receipt = receipt
        else:
            proof = seal.terminal_record(raw, job, seal.retained.ENTRY)
            proof['exit_receipt'] = receipt; cap['scheduler_proofs'][role] = proof
    return cap, queue, captured, capture_receipt, receipts


def data_fixture():
    return dict(status='placement_expanded_corpus_and_two_epoch_recipe_prepared_not_parent_or_fit_registered',
        optimizer_updates=0, student_fit_registered=False, originals_preserved_object_exact=True,
        inherited_split_tags_preserved=True, validation_file_reused_unchanged=True, id_overlap=0, group_overlap=0,
        source_loss_mass={s: float(v) for s, v in seal.masses().items()},
        source_unique_records=dict.fromkeys(seal.masses(), 2))


class RetainedPlacementSealTests(unittest.TestCase):
    def test_actual_format_all_four_successful_gpu_phases_are_required(self):
        cap, queue, captured, receipt, receipts = phases()
        exits, paths = seal.phase_proofs(cap, queue, captured, receipt, receipts.__getitem__)
        self.assertEqual(set(exits), {'prepare', 'fit', 'audit', 'capture'})
        self.assertEqual(len(paths), 4)

    def test_live_failed_cpu_only_or_other_phase_is_rejected(self):
        for mutation in ('live', 'failed', 'cpu_only', 'command', 'queue', 'role', 'missing'):
            cap, queue, captured, receipt, receipts = phases(); proof = cap['scheduler_proofs']['fit']
            if mutation == 'live': proof['raw'] = proof['raw'].replace('JobState=COMPLETED', 'JobState=RUNNING')
            elif mutation == 'failed': proof['exit_receipt']['exit_code'] = 1
            elif mutation == 'cpu_only': proof['raw'] = proof['raw'].replace('gres/gpu=1', 'gres/gpu=0')
            elif mutation == 'command': proof['raw'] = proof['raw'].replace(str(seal.retained.ENTRY.resolve()), '/unrelated-entry')
            elif mutation == 'queue': queue['fit_job'] = '999'
            elif mutation == 'role': proof['exit_receipt']['phase'] = 'prepare'
            else: del cap['scheduler_proofs']['prepare']
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                seal.phase_proofs(cap, queue, captured, receipt, receipts.__getitem__)

    def test_capture_own_terminal_must_match_held_job_and_exit(self):
        for mutation in ('job', 'phase', 'failed', 'raw'):
            cap, queue, captured, receipt, receipts = phases()
            if mutation == 'job': receipt['job_id'] = '999'
            elif mutation == 'phase': receipt['phase'] = 'fit'
            elif mutation == 'failed': receipt['exit_code'] = 1
            else: captured['raw'] = captured['raw'].replace('JobState=COMPLETED', 'JobState=FAILED')
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                seal.phase_proofs(cap, queue, captured, receipt, receipts.__getitem__)

    def test_all57_positive_original_pools_and_unchanged_mass_are_required(self):
        original = dict(status='placement_all_sources_including_targeted_win_special_prepared_not_fit')
        seal.require_data(data_fixture(), original)
        for mutation in ('missing', 'zero', 'mass', 'overlap', 'expanded_val', 'already_fit'):
            data = data_fixture(); source = next(iter(data['source_unique_records']))
            if mutation == 'missing': del data['source_unique_records'][source]
            elif mutation == 'zero': data['source_unique_records'][source] = 0
            elif mutation == 'mass': data['source_loss_mass'][source] *= 2
            elif mutation == 'overlap': data['group_overlap'] = 1
            elif mutation == 'expanded_val': data['validation_file_reused_unchanged'] = False
            else: data['student_fit_registered'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): seal.require_data(data, original)

    def test_another_parent_lineage_cannot_enter_the_producer(self):
        with self.assertRaisesRegex(ValueError, 'fixed retained3B'):
            seal.parent_artifacts(Path('/unrelated-parent/fit-plan/registered.json'))

    def test_missing_real_parent_cannot_leave_a_partial_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'not-created'
            with patch.object(seal, 'resource', return_value='fixture'), \
                    patch.object(seal, 'parent_artifacts', side_effect=FileNotFoundError('actual parent missing')):
                with self.assertRaises(FileNotFoundError): seal.seal(seal.PARENT, Path('/unused'), Path('/unused'), target)
            self.assertFalse(target.exists())

    def test_existing_seal_is_not_replaced_or_requalified(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory); sentinel = target/'keep'; sentinel.write_text('original')
            with patch.object(seal, 'resource', return_value='fixture'), patch.object(seal, 'parent_artifacts') as parent:
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    seal.seal(seal.PARENT, Path('/unused'), Path('/unused'), target)
                parent.assert_not_called()
            self.assertEqual(sentinel.read_text(), 'original')


if __name__ == '__main__': unittest.main()
