"""Run the analytic distance cases on real CUDA, including breaks and conflicts."""

import json
import unittest
from unittest.mock import patch

import torch

from pipeline.distance import Distance
from pipeline.tests import test_pipeline as cases


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class CudaDistanceTests(cases.DistanceTests):
    def metric(self, **overrides):
        return super().metric(device="cuda:0", **overrides)

    def test_numeric_state_stays_on_cuda_and_output_is_json(self):
        metric = self.metric(smoothing_tau_s=.12)
        for index in range(5):
            contributions = metric.update(cases.row(index, [cases.person(index * .1)]))
            json.dumps(contributions, allow_nan=False)
        state = next(iter(metric.previous.values()))
        self.assertEqual(state['filtered'].device.type, 'cuda')
        for name in ('distance', 'tracked_duration_s', 'measured_duration_s', 'valid_pairs'):
            self.assertEqual(next(iter(metric.rows.values()))[name].device.type, 'cuda')
        json.dumps(metric.result(), allow_nan=False)

    def test_cuda_matches_reference_through_gaps_cuts_and_smoothing(self):
        cpu = cases.DistanceTests().metric(smoothing_tau_s=.12, deadband_m=.025)
        gpu = self.metric(smoothing_tau_s=.12, deadband_m=.025)
        for index in range(100):
            people = [] if index % 19 == 0 else [cases.person(None if index % 13 == 0 else index * .05, name='A')]
            frame = cases.row(index, people, segment=index // 25, included=index % 17 != 0)
            expected, actual = cpu.update(frame), gpu.update(frame)
            for a, b in zip(expected, actual, strict=True):
                self.assertEqual(a['reason'], b['reason'])
                if a['distance_m'] is not None:
                    self.assertAlmostEqual(a['distance_m'], b['distance_m'], places=12)
        a, b = cpu.result()['players'], gpu.result()['players']
        for first, second in zip(a, b, strict=True):
            for key in ('observed_distance_m', 'tracked_duration_s', 'measured_duration_s', 'coverage_ratio'):
                self.assertAlmostEqual(first[key], second[key], places=12)
            self.assertEqual(first['warnings'], second['warnings'])


class CudaPolicyTests(unittest.TestCase):
    def test_missing_cuda_never_uses_cpu(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'refusing CPU fallback'):
                Distance({'device': 'cuda:0'}, 'test')


if __name__ == '__main__':
    unittest.main()
