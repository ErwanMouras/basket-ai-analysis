import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.progress import Progress


class ProgressTests(unittest.TestCase):
    def test_video_completion_leaves_statistics_and_health_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Progress(directory, 'test')
            try:
                progress.update('perception', 100, 100, frames=100, total_frames=100)
                path = Path(directory) / 'progress.json'
                self.assertLess(json.loads(path.read_text())['overall_percent'], 100)
                progress.update('statistics', 50, 100)
                state = json.loads(path.read_text())
                self.assertEqual(state['stage_percent'], 50)
                self.assertEqual(state['frames'], 100)
                self.assertLess(state['overall_percent'], 100)
                progress.update('health', 100, 100)
                self.assertLess(json.loads(path.read_text())['overall_percent'], 100)
                progress.update('health', 100, 100, status='completed')
                self.assertEqual(json.loads(path.read_text())['overall_percent'], 100)
            finally:
                progress.close()

    def test_health_failure_keeps_run_failed_and_progress_below_100(self):
        from pipeline.tests import test_pipeline as cases
        fixture = cases.IntegrationTests()
        fixture.setUp()
        try:
            with patch('mlops.health.build_health', side_effect=RuntimeError('integrity failure')):
                with self.assertRaisesRegex(RuntimeError, 'integrity failure'):
                    fixture.run_video('health-failed')
            output = fixture.root / 'health-failed'
            state = json.loads((output / 'progress.json').read_text())
            self.assertEqual(state['stage'], 'health')
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['frames'], 8)
            self.assertLess(state['overall_percent'], 100)
            self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'failed')
        finally:
            fixture.doCleanups()

    def test_failure_preserves_stage_and_never_reports_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Progress(directory, 'test', identity=False)
            try:
                progress.update('statistics', 42, 100, status='failed', error='CUDA failure')
                state = json.loads((Path(directory) / 'progress.json').read_text())
                self.assertEqual(state['stage'], 'statistics')
                self.assertEqual(state['error'], 'CUDA failure')
                self.assertEqual(state['status'], 'failed')
                self.assertLess(state['overall_percent'], 100)
                self.assertNotIn('identity_ocr', state['phase_weights'])
            finally:
                progress.close()
