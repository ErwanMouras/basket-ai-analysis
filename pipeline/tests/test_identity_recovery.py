import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.identity_recovery import merge_reads, recover_masked_reads, sample_frames


class RecoveryTests(unittest.TestCase):
    def test_missing_cuda_fails_before_reading_or_creating_artifacts(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict('sys.modules', {'torch': torch}):
            with self.assertRaisesRegex(RuntimeError, 'CPU fallback is disabled'):
                recover_masked_reads('/missing/run', '/missing/output')

    def test_sampling_spans_the_track_and_preserves_real_frame_indices(self):
        self.assertEqual(sample_frames([5, 7, 11], 12), {5, 7, 11})
        self.assertEqual(sample_frames(list(range(10, 110)), 3), {10, 60, 109})

    def test_multiple_crops_cannot_add_votes_to_an_existing_source_frame(self):
        old = {'decoder': 'digits_only_v4', 'segment_id': 1, 'track_id': 2,
               'frame_index': 50, 'raw_number': '27', 'confidence': .92}
        improved = {**old, 'confidence': .99}
        legacy = {**old, 'decoder': 'digits_only_v2'}
        self.assertEqual(merge_reads([legacy, old], [improved]), [legacy, improved])
        self.assertEqual(merge_reads([old], [{**improved, 'confidence': .8}]), [old])
        extra = {**improved, 'frame_index': 52}
        self.assertEqual(merge_reads([old], [improved, extra]), [improved, extra])
        self.assertEqual(merge_reads([], [improved, old, extra]), [improved, extra])
