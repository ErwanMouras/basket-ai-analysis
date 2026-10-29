import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from pipeline.monitor import read_progress
from pipeline.progress import Progress


class ProgressTests(unittest.TestCase):
    def test_heartbeat_does_not_claim_work_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Progress(directory, 'test')
            try:
                progress.update('perception', 30, 100, frames=30, total_frames=100)
                advanced = progress.state['last_progress_at']
                progress.last_progress -= 100
                progress._write()
                self.assertEqual(progress.state['last_progress_at'], advanced)
                self.assertGreaterEqual(progress.state['seconds_since_progress'], 100)
                progress.update('perception', 60, 100, frames=60)
                self.assertLess(progress.state['seconds_since_progress'], 1)
            finally:
                progress.close()

    def test_monitor_hides_stale_speed_while_process_is_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            state = {'status': 'running', 'pid': 123, 'stage': 'perception',
                     'last_progress_at': (datetime.now(UTC) - timedelta(minutes=3)).isoformat(),
                     'throughput_fps': 11.2, 'perception_remaining_s': 200}
            path.write_text(json.dumps(state))
            with patch('pipeline.monitor.os.kill'):
                shown = read_progress(directory)
            self.assertEqual(shown['status'], 'stalled')
            self.assertIsNone(shown['throughput_fps'])
            self.assertIsNone(shown['perception_remaining_s'])
            self.assertIn('Aucune progression', shown['error'])
            state['last_progress_at'] = datetime.now(UTC).isoformat()
            path.write_text(json.dumps(state))
            with patch('pipeline.monitor.os.kill'):
                self.assertEqual(read_progress(directory)['status'], 'running')

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
            with (
                patch('mlops.health.build_health', side_effect=RuntimeError('integrity failure')),
                self.assertRaisesRegex(RuntimeError, 'integrity failure'),
            ):
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
