import unittest

import numpy as np

from pipeline.jersey_votes import sample_team
from training.jersey.colors import TeamColors, dominant_color
from training.segmentation.masks import encode


class SegmentedColorTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((100, 100, 3), (30, 60, 220), dtype=np.uint8)
        self.foreground = np.zeros((100, 100), dtype=bool)
        self.foreground[:, 26:74] = True
        self.rgb[self.foreground] = (235, 235, 235)

    def test_only_player_pixels_vote_even_when_background_changes(self):
        self.assertIsNone(dominant_color(self.rgb))
        self.assertEqual(dominant_color(self.rgb, self.foreground), ('light', 0.0))
        self.rgb[~self.foreground] = (220, 30, 30)
        self.assertEqual(dominant_color(self.rgb, self.foreground), ('light', 0.0))

    def test_missing_or_tiny_foreground_cannot_become_a_uniform_color(self):
        self.assertIsNone(dominant_color(self.rgb, np.zeros_like(self.foreground)))
        tiny = np.zeros_like(self.foreground)
        tiny[40:44, 40:44] = True
        self.assertIsNone(dominant_color(self.rgb, tiny))
        with self.assertRaises(ValueError):
            dominant_color(self.rgb, tiny[:20])

    def test_completed_run_sampling_and_live_groups_both_use_the_mask(self):
        signatures = {'white': ('light', 0.0), 'blue': ('hue', 115.0)}
        segmentation = {'status': 'ok', 'mask': encode(self.foreground, [0, 0, 100, 100])}
        self.assertEqual(sample_team(self.rgb[..., ::-1].copy(), [0, 0, 100, 100],
                                     signatures, segmentation), 'white')
        groups = TeamColors()
        state = {'last_color_sample': -100, 'color_votes': TeamColors.new_votes(), 'team_group': None}
        crop = {'image': self.rgb, 'color_image': self.rgb, 'color_mask': self.foreground, 'overlap': 0}
        for time in (0, .4, .8):
            groups.observe(state, crop, time)
        self.assertEqual(state['team_group'], 'team_1')
        self.assertEqual(groups.groups, [('light', 0.0)])
