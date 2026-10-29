"""Identity continuity must stop at cuts, crossings and contradictory evidence."""

import tempfile
import unittest
from contextlib import ExitStack

import torch

from pipeline.cuda.core import CudaOnly, tensor
from pipeline.cuda.recovery import read_rank, uniform_team
from pipeline.cuda.refined_identity import (
    match_bodies,
    propagate_motion,
    refinement_episodes,
)
from pipeline.numeric import cuda_numbers
from pipeline.progress import Progress


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class WholeRefinementTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        for ctx in (
            torch.inference_mode(),
            torch.device("cuda:0"),
            cuda_numbers("cuda:0"),
            CudaOnly(),
        ):
            self.context.enter_context(ctx)
        self.device = "cuda:0"

    def person(self, track=1, x=0):
        return {"track_id": track, "role": "player", "bbox": tensor([x, 0, x + 10, 30])}

    def row(self, index, track=1, cut=False, x=0):
        return {
            "frame_index": index,
            "segment_id": 0,
            "timestamp_s": tensor(index * 0.04),
            "scene_cut": cut,
            "persons": [self.person(track, x)],
        }

    def test_invalid_high_confidence_read_cannot_erase_valid_number(self):
        valid = {"number": "24", "confidence": 0.86}
        invalid = {"number": None, "confidence": 0.999}
        self.assertGreater(
            float(read_rank(valid, self.device)), float(read_rank(invalid, self.device))
        )

    def test_uniform_recovery_rejects_middle_grey_and_foreground_absence(self):
        tags = {"dark": "sac", "light": "sas"}
        box = tensor([0, 0, 100, 200])
        points = torch.zeros((17, 2), device=self.device)
        valid = torch.zeros(17, device=self.device, dtype=torch.bool)
        mask = torch.ones((200, 100), device=self.device, dtype=torch.bool)
        for brightness, expected in ((30, "sac"), (225, "sas"), (135, None)):
            image = torch.full(
                (3, 200, 100), brightness, device=self.device, dtype=torch.uint8
            )
            team, _ = uniform_team(image, box, points, valid, mask, tags)
            self.assertEqual(team, expected)
        self.assertIsNone(uniform_team(image, box, points, valid, ~mask, tags)[0])

    def test_gap_is_joined_only_with_safe_geometry_and_color(self):
        rows = {i: self.row(i) for i in (0, 1, 3, 4)}
        groups = refinement_episodes(
            list(rows), {i: "a" for i in rows}, rows, (0, 1), self.device, {}
        )
        self.assertEqual(groups, [[0, 1, 3, 4]])
        rows[3]["persons"][0]["bbox"] += tensor([100, 0, 100, 0])
        self.assertEqual(
            refinement_episodes(
                list(rows), {i: "a" for i in rows}, rows, (0, 1), self.device, {}
            ),
            [[0, 1], [3, 4]],
        )

    def test_ambiguous_duplicate_bodies_have_no_motion_link(self):
        before = [self.person()]
        after = [self.person(2), self.person(3)]
        self.assertEqual(match_bodies(before, after, self.device), [])

    def propagate(self, rows, direct):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Progress(tmp, "test")
            try:
                identities = direct, {}, {}, {}, set()
                propagate_motion(rows, identities, self.device, progress)
                return direct
            finally:
                progress.close()

    def test_motion_fills_new_source_id_but_does_not_cross_cut(self):
        p = {"team_id": "a", "number": "1"}
        rows = {i: self.row(i, track=i + 1) for i in range(3)}
        direct = self.propagate(rows, {(0, 1, 0): p})
        self.assertEqual(direct[(0, 2, 1)]["number"], "1")
        rows[1]["scene_cut"] = True
        direct = self.propagate(rows, {(0, 1, 0): p})
        self.assertNotIn((0, 2, 1), direct)

    def test_opposing_bidirectional_anchors_leave_middle_unknown(self):
        rows = {i: self.row(i, track=i + 1) for i in range(3)}
        direct = self.propagate(
            rows,
            {
                (0, 1, 0): {"team_id": "a", "number": "1"},
                (0, 3, 2): {"team_id": "a", "number": "2"},
            },
        )
        self.assertNotIn((0, 2, 1), direct)

    def test_sustained_number_change_splits_same_color_track(self):
        rows = {i: self.row(i) for i in range(20)}
        reads = {
            i: {
                "frame_index": i,
                "timestamp_seconds": i * 0.04,
                "confidence": tensor(0.99),
                "number": "1" if i < 10 else "24",
            }
            for i in (0, 2, 4, 10, 12, 14)
        }
        groups = refinement_episodes(
            list(rows), {i: "a" for i in rows}, rows, (0, 1), self.device, reads
        )
        self.assertEqual(groups, [list(range(10)), list(range(10, 20))])


if __name__ == "__main__":
    unittest.main()
