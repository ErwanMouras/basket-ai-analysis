"""Analytic GPU contracts for the native path; all numerical stages guarded."""

import unittest
from contextlib import ExitStack

import torch

from pipeline.config import load_config
from pipeline.cuda.core import CudaOnly, export, project, tensor
from pipeline.cuda.geometry import (
    Court,
    _hull_reference,
    _optical_flow_reference,
    area,
    dlt,
    hull,
    optical_flow,
    polygon_distance,
    ransac,
)
from pipeline.cuda.identity import split_episodes
from pipeline.cuda.models import encode_mask
from pipeline.cuda.tracking import (
    Tracker,
    _linear_assignment_reference,
    linear_assignment,
)
from pipeline.cuda.vision import candidate, color_distance, dominant_color
from pipeline.distance import Distance
from pipeline.numeric import CudaCounter, cuda_numbers


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class NativeCudaTests(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0"
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.inference_mode())
        self.stack.enter_context(torch.device(self.device))
        self.stack.enter_context(cuda_numbers(self.device))
        self.guard = self.stack.enter_context(CudaOnly())

    def test_assignment_finds_global_optimum_not_greedy(self):
        matches, left, right = linear_assignment(tensor([[0.1, 0.2], [0.11, 0.9]]), 1.0)
        self.assertEqual(set(matches), {(0, 1), (1, 0)})
        self.assertEqual(left, [])
        self.assertEqual(right, [])

    def test_fused_assignment_and_hull_match_cuda_reference(self):
        torch.manual_seed(19)
        for n, m in ((1, 3), (4, 2), (11, 15), (30, 20)):
            cost = torch.rand((n, m), device=self.device, dtype=torch.float64)
            cost = torch.where(cost > 0.85, torch.inf, cost)
            self.assertEqual(
                set(linear_assignment(cost, 0.6)[0]),
                set(_linear_assignment_reference(cost, 0.6)[0]),
            )
        for count in (2, 18, 150):
            points = torch.rand((count, 2), device=self.device, dtype=torch.float64)
            self.assertTrue(torch.equal(hull(points), _hull_reference(points)))

    def test_packed_export_preserves_shapes_types_aliases_and_empty_values(self):
        values = torch.arange(6, device=self.device).reshape(2, 3)
        payload = {
            "a": values,
            "b": values,
            "c": (tensor(2.5), torch.tensor(True, device=self.device)),
            "empty": torch.empty((2, 0), device=self.device),
            "none": None,
        }
        self.assertEqual(
            export(payload),
            {
                "a": [[0, 1, 2], [3, 4, 5]],
                "b": [[0, 1, 2], [3, 4, 5]],
                "c": [2.5, True],
                "empty": [[], []],
                "none": None,
            },
        )

    def test_stream_dependencies_and_worker_cpu_guard(self):
        from pipeline.cuda.streams import ModelsInFlight

        runner = ModelsInFlight(self.device)
        self.addCleanup(runner.close)
        value = torch.arange(50, device=self.device, dtype=torch.float64)
        job = runner.submit("pose", lambda x: x.square() + 2, value)
        result = runner.result(job)
        self.assertTrue(torch.equal(result, value.square() + 2))
        job = runner.submit("ball", lambda: torch.ones(3, device="cpu"))
        with self.assertRaisesRegex(RuntimeError, "GPU-only violation"):
            runner.result(job)

    def test_captured_model_uses_new_inputs_and_protects_static_buffers(self):
        from pipeline.cuda.graphs import CapturedCall

        captured = CapturedCall(lambda x: (x.square(), x + 1), tensor([1.0, 2.0]))
        first = captured(tensor([3.0, 4.0]))
        first[0].zero_()
        second = captured(tensor([5.0, 6.0]))
        self.assertTrue(torch.equal(second[0], tensor([25.0, 36.0])))
        self.assertTrue(torch.equal(second[1], tensor([6.0, 7.0])))

    def test_assignment_respects_cost_limit_and_rectangular_empty_cases(self):
        matches, left, right = linear_assignment(
            tensor([[0.1, 0.9, 0.8], [0.9, 0.9, 0.9]]), 0.5
        )
        self.assertEqual(matches, [(0, 0)])
        self.assertEqual(left, [1])
        self.assertEqual(right, [1, 2])
        self.assertEqual(
            linear_assignment(torch.empty((0, 2), device=self.device), 0.5),
            ([], [], [0, 1]),
        )

    def test_assignment_many_ties_and_gated_rows_across_warps(self):
        for n, m in ((33, 7), (65, 33), (12, 65)):
            cost = torch.zeros((n, m), device=self.device, dtype=torch.float64)
            for _ in range(8):
                matches, left, right = linear_assignment(cost, 0.5)
                self.assertEqual(len(matches), min(n, m))
                self.assertEqual(len(left), n - len(matches))
                self.assertEqual(len(right), m - len(matches))
                self.assertEqual(len({r for r, _ in matches}), len(matches))
                self.assertEqual(len({c for _, c in matches}), len(matches))
            cost.fill_(torch.inf)
            self.assertEqual(linear_assignment(cost, 0.5), ([], list(range(n)), list(range(m))))

    def test_assignment_invalid_infinite_path_raises_instead_of_spinning(self):
        with self.assertRaisesRegex(RuntimeError, 'invalid or cyclic'):
            linear_assignment(tensor([[-torch.inf]]), 0.5)

    def test_tracker_roles_cuts_and_missing_detection(self):
        config = load_config()["tracking"]
        config.update(tracker="bytetrack", gmc_method="none")
        tracker = Tracker(config, 25, self.device)
        rows = tensor([[20, 20, 50, 100, 0.99, 2], [100, 20, 130, 100, 0.99, 3]])
        first = tracker.update(rows, 0)
        second = tracker.update(rows, 1)
        self.assertEqual(first, second)
        swap = rows.clone()
        swap[:, 5] = swap[:, 5].flip(0)
        self.assertEqual(tracker.update(swap, 2), [None, None])
        self.assertNotEqual(tracker.update(rows, 3, cut=True), first)

    def test_normalized_dlt_recovers_projective_mapping(self):
        points = tensor([[0, 0], [10, 0], [10, 10], [0, 10], [3, 7], [9, 2]])
        truth = tensor([[2, 0.1, 30], [-0.2, 3, 40], [0.001, 0.002, 1]])
        result = dlt(points, project(truth, points))
        self.assertTrue(bool(torch.allclose(result, truth, atol=1e-8)))

    def test_ransac_rejects_outliers(self):
        torch.manual_seed(7)
        first = torch.rand((30, 2), device=self.device, dtype=torch.float64) * 20
        truth = tensor([[20, 3, 140], [-1, 25, 160], [0.001, 0.002, 1]])
        second = project(truth, first)
        second[-5:] += 200
        result, inliers = ransac(first, second, 1.0, samples=512)
        self.assertIsNotNone(result)
        self.assertEqual(int(inliers.sum()), 25)
        self.assertTrue(
            bool(torch.allclose(project(result, first[:-5]), second[:-5], atol=1e-6))
        )

    def test_hull_area_and_signed_support_distance(self):
        polygon = hull(tensor([[0, 0], [10, 0], [5, 5], [10, 10], [0, 10]]))
        self.assertAlmostEqual(float(area(polygon)), 100)
        distance = polygon_distance(tensor([[5, 5], [15, 5]]), polygon)
        self.assertTrue(bool(torch.allclose(distance, tensor([5, -5]))))

    def test_pyramidal_lk_tracks_known_translation(self):
        torch.manual_seed(11)
        image = torch.rand((160, 192), device=self.device) * 255
        image = torch.nn.functional.avg_pool2d(image[None, None], 3, 1, 1)[0, 0]
        following = image.roll((2, 3), (0, 1))
        points = tensor(
            [[60, 60], [90, 70], [120, 100], [70, 110]], dtype=torch.float32
        )
        tracked, valid = optical_flow(image, following, points)
        self.assertTrue(bool(valid.all()))
        self.assertLess(
            float(
                torch.linalg.vector_norm(
                    tracked - points - tensor([3, 2], dtype=torch.float32), dim=1
                ).amax()
            ),
            0.5,
        )
        expected, good = _optical_flow_reference(image, following, points)
        self.assertTrue(torch.equal(valid, good))
        self.assertLess(float((tracked - expected).abs().amax()), 0.002)

    def test_batched_projection_and_floor_mask_preserve_reference(self):
        config = load_config()["court"]
        court = Court(config, None, self.device)
        image = torch.empty((3, 540, 960), device=self.device)
        rows = tensor(
            [
                [100, 100, 180, 250, 0.9, 2],
                [20, 10, 100, 539, 0.9, 2],
                [500, 100, 550, 300, 0.1, 2],
            ],
            dtype=torch.float32,
        )
        pose = torch.zeros((3, 17, 2), device=self.device)
        valid = torch.zeros((3, 17), device=self.device, dtype=torch.bool)
        pose[0, 15:17] = tensor([[130, 240], [140, 240]], dtype=torch.float32)
        valid[0, 15:17] = True
        for state in (
            None,
            {
                "matrix": tensor([[0.03, 0, 0], [0, 0.03, 0], [0, 0, 1]]),
                "support": tensor([[0, 0], [28, 0], [28, 15], [0, 15]]),
            },
        ):
            court.state = state
            self.assertTrue(
                torch.equal(
                    court._mask(image, rows, (270, 480)),
                    court._mask_reference(image, rows, (270, 480)),
                )
            )
            self.assertEqual(
                export(court.positions(rows, pose, valid, {}, image.shape[-2:])),
                export(
                    court._positions_reference(rows, pose, valid, {}, image.shape[-2:])
                ),
            )

    def test_court_fit_preserves_source_pixel_error_bounds(self):
        config = load_config()["court"]
        court = Court(config, None, self.device)
        truth = tensor([[35, 15, 150], [-2, 40, 150], [0.001, 0.003, 1]])
        pixels = project(truth, court.landmarks)
        keypoints = torch.cat(
            (
                pixels,
                torch.full((18, 1), 0.95, device=self.device, dtype=torch.float64),
            ),
            1,
        )
        fit, report = court.fit(keypoints, (1080, 1920))
        self.assertIsNotNone(fit, report)
        self.assertLess(float(report["reprojection_rmse_px"]), 1e-5)
        self.assertTrue(
            bool(
                torch.allclose(
                    project(fit["matrix"], pixels), court.landmarks, atol=1e-5
                )
            )
        )

    def test_color_signatures_and_unreadable_mask(self):
        light = torch.full((3, 40, 40), 255, device=self.device)
        dark = torch.zeros((3, 40, 40), device=self.device)
        a, b = dominant_color(light), dominant_color(dark)
        self.assertEqual(a[0], "light")
        self.assertEqual(b[0], "dark")
        self.assertEqual(float(color_distance(a, b)), 180)
        self.assertIsNone(
            dominant_color(
                light, torch.zeros((40, 40), device=self.device, dtype=torch.bool)
            )
        )

    def test_mask_rle_round_trip(self):
        mask = torch.zeros((20, 30), device=self.device, dtype=torch.bool)
        mask[5:12, 6:19] = True
        payload, nonempty = encode_mask(mask, tensor([2.1, 3.2, 24.7, 17.3]))
        a, b, c, d = payload["roi"].tolist()
        decoded = (
            torch.repeat_interleave(
                torch.arange(len(payload["counts"]), device=self.device).remainder(2),
                payload["counts"],
            )
            .bool()
            .reshape(d - b, c - a)
        )
        self.assertTrue(nonempty)
        self.assertTrue(bool((decoded == mask[b:d, a:c]).all()))

    def test_ocr_crop_and_mask_neutralization_are_cuda(self):
        torch.manual_seed(3)
        image = torch.rand((3, 300, 200), device=self.device) * 255
        rows = tensor([[0, 0, 200, 300, 0.99, 2]], dtype=torch.float32)
        masks = torch.ones((1, 300, 200), device=self.device, dtype=torch.bool)
        config = load_config()["jersey"]
        config["mask_policy"] = "background_neutral"
        crop = candidate(
            image,
            rows,
            0,
            torch.zeros((17, 2), device=self.device),
            torch.zeros(17, device=self.device, dtype=torch.bool),
            masks,
            config,
            "upper",
        )
        self.assertIsNotNone(crop)
        self.assertEqual(crop["image"].device.type, "cuda")
        self.assertEqual(tuple(crop["image"].shape), (3, 32, 128))
        self.assertEqual(crop["mask_audit"]["status"], "applied")

    def test_color_switches_and_gaps_split_episodes(self):
        frames = list(range(9))
        colors = {i: "a" if i < 3 else "b" for i in frames}
        self.assertEqual(
            split_episodes(frames, colors, self.device),
            [list(range(3)), list(range(3, 9))],
        )
        self.assertEqual(
            split_episodes([0, 1, 2, 5, 6, 7], {}, self.device), [[0, 1, 2], [5, 6, 7]]
        )

    def test_identity_and_exclusion_counters_are_cuda(self):
        counter = CudaCounter(["a", "a", "b"], device=self.device)
        self.assertTrue(all(value.is_cuda for value in counter.values()))
        self.assertEqual(counter.most_common(1)[0][0], "a")
        from pipeline.tests.test_pipeline import person, row

        metric = Distance(
            {
                **load_config()["distance"],
                "device": self.device,
                "smoothing_tau_s": 0,
                "deadband_m": 0,
            },
            "test",
        )
        for i, position in enumerate((0, 0.1, None, 0.3, 0.4)):
            metric.update(row(i, [person(position)]))
        self.assertTrue(
            all(
                value.is_cuda
                for item in metric.rows.values()
                for value in item["excluded_intervals"].values()
            )
        )
        self.assertAlmostEqual(
            metric.result()["players"][0]["observed_distance_m"], 0.2
        )

    def test_spatial_handoff_geometry_and_votes_stay_cuda(self):
        from pipeline.spatial_identity import find_spatial_handoffs

        rows = {}
        for frame in range(8):
            tid = 1 if frame < 4 else 2
            rows[frame] = {
                "frame_index": frame,
                "segment_id": 0,
                "timestamp_s": tensor(frame) / 10,
                "scene_cut": False,
                "included": True,
                "persons": [
                    {
                        "role": "player",
                        "track_id": tid,
                        "bbox": tensor([100, 100, 150, 250])
                        + tensor([frame, 0, frame, 0]),
                        "position_m": tensor([frame, 0]) / 10,
                    }
                ],
            }
        player = {"team_id": "a", "number": "1", "player_name": "Alice"}
        assigned = {(0, 1): player}
        votes = {}
        for tid, frames in ((1, range(4)), (2, range(4, 8))):
            votes[(0, tid)] = {
                "team_id": "a",
                "raw": {frame: "a" for frame in frames},
                "counts": CudaCounter({"a": 4}, device=self.device),
            }
        handoffs = find_spatial_handoffs(
            None, None, assigned, votes, rows=rows, all_reads=[]
        )
        self.assertIn((0, 2, 4), handoffs)
        self.assertEqual(handoffs[(0, 2, 4)]["player"]["player_name"], "Alice")
        self.assertTrue(handoffs[(0, 2, 4)]["distance_m"].is_cuda)
        votes[(0, 2)] = {
            "team_id": "b",
            "raw": {frame: "b" for frame in range(4, 8)},
            "counts": CudaCounter({"b": 4}, device=self.device),
        }
        self.assertFalse(
            find_spatial_handoffs(None, None, assigned, votes, rows=rows, all_reads=[])
        )

    def test_simultaneous_tracks_cannot_merge_into_one_roster_identity(self):
        from pipeline.cuda.identity import collision_frames

        player = {"team_id": "a", "number": "1", "player_name": "Alice"}
        rows = {
            0: {
                "frame_index": 0,
                "segment_id": 0,
                "persons": [
                    {"role": "player", "track_id": 1},
                    {"role": "player", "track_id": 2},
                ],
            }
        }
        direct = {(0, 1, 0): player, (0, 2, 0): player}
        self.assertEqual(
            collision_frames(rows, direct, {}, {}, self.device),
            {(0, (0, 1)), (0, (0, 2))},
        )


class GuardTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_host_arithmetic_fails_and_exports_do_not_authorize_it(self):
        cpu = torch.ones(2, device="cpu")
        gpu = torch.ones(2, device="cuda:0")
        with CudaOnly():
            self.assertEqual(gpu.tolist(), [1, 1])
            with self.assertRaisesRegex(RuntimeError, "GPU-only violation"):
                cpu + 1
            with self.assertRaisesRegex(RuntimeError, "GPU-only violation"):
                gpu.cpu() + 1
            with self.assertRaisesRegex(RuntimeError, "GPU-only violation"):
                torch.zeros(2, device="cpu")
