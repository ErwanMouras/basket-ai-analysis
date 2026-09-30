import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from pipeline.identity import replay_artifacts, track_assignments
from pipeline.jersey_votes import _selected_frames, build_track_votes, majority, split_color_episodes
from training.jersey.roster import Roster


class IdentityReplayTests(unittest.TestCase):
    def test_color_switch_splits_reused_track_id(self):
        frames = list(range(12))
        colors = {frame: 'dark' if frame < 6 else 'gold' for frame in frames}
        self.assertEqual(split_color_episodes(frames, colors),
                         [list(range(6)), list(range(6, 12))])

    def test_ocr_stops_after_episode_majority_and_color_continues(self):
        roster = Roster({'schema_version': 1, 'teams': [
            {'team_id': 'dark', 'name': 'Dark', 'uniform_color': '#202020',
             'players': [{'number': '10', 'name': 'Alice'}]},
            {'team_id': 'gold', 'name': 'Gold', 'uniform_color': '#FDB927',
             'players': [{'number': '10', 'name': 'Bob'}]}]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, observations, reads, votes = (root / name for name in
                ('sample.avi', 'observations.jsonl', 'reads.jsonl', 'team_votes.json'))
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (128, 128))
            with observations.open('w') as output:
                for frame in range(12):
                    writer.write(np.full((128, 128, 3), 32, dtype=np.uint8))
                    output.write(json.dumps({'frame_index': frame, 'timestamp_s': frame / 10,
                        'segment_id': 0, 'persons': [{'role': 'player', 'track_id': 1,
                        'bbox': [0, 0, 128, 128], 'confidence': .9}]}) + '\n')
            writer.release()
            reads.write_text('')

            class FakeReader:
                def __init__(self, config):
                    self.provenance = {'decoder': 'test'}

                def read(self, crops):
                    return [{'number': '10', 'text': '10', 'confidence': .95}
                            for _ in crops]

            colors = ['dark'] * 6 + ['gold'] * 6
            crop = {'image': np.zeros((32, 128, 3), dtype=np.uint8),
                    'bbox': [10, 10, 40, 40], 'region': 'bbox_number', 'quality': .9}
            with patch('pipeline.jersey_votes.sample_team', side_effect=colors), \
                    patch('training.jersey.crops.candidate', return_value=crop), \
                    patch('training.jersey.reader.ParseqReader', FakeReader):
                build_track_votes(observations, reads, video, roster, 'test', votes)
            document = json.loads(votes.read_text())
            item = document['tracks']['0:1']
            self.assertEqual(item['counts'], {'dark': 6, 'gold': 6})
            self.assertEqual([episode['confirmation_frame'] for episode in item['episodes']],
                             [4, 10])
            self.assertEqual([json.loads(line)['frame_index'] for line in reads.read_text().splitlines()],
                             [0, 2, 4, 6, 8, 10])

    def test_number_sampling_spans_the_whole_track(self):
        with tempfile.TemporaryDirectory() as directory:
            observations = Path(directory) / 'observations.jsonl'
            with observations.open('w') as output:
                for frame in range(100):
                    output.write(json.dumps({'frame_index': frame, 'timestamp_s': frame / 30,
                        'segment_id': 0, 'persons': [{'role': 'player', 'track_id': 7}]}) + '\n')
            selected = sorted(frame for segment, track, frame in _selected_frames(observations))
            self.assertGreaterEqual(len(selected), 12)
            self.assertEqual(selected[0], 0)
            self.assertGreaterEqual(selected[-1], 90)

    def test_team_color_majority_counts_every_visible_frame(self):
        roster = Roster({'schema_version': 1, 'teams': [
            {'team_id': 'dark', 'name': 'Dark', 'uniform_color': '#202020',
             'players': [{'number': '10', 'name': 'Alice'}]},
            {'team_id': 'gold', 'name': 'Gold', 'uniform_color': '#FDB927',
             'players': [{'number': '10', 'name': 'Bob'}]},
        ]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, observations, reads, votes = (root / name for name in
                ('sample.avi', 'observations.jsonl', 'reads.jsonl', 'team_votes.json'))
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (128, 128))
            with observations.open('w') as output:
                for frame in range(10):
                    color = (39, 185, 253) if frame in (3, 4, 5) else (32, 32, 32)
                    writer.write(np.full((128, 128, 3), color, dtype=np.uint8))
                    output.write(json.dumps({'frame_index': frame, 'timestamp_s': frame / 10,
                        'segment_id': 0, 'persons': [{'role': 'player', 'track_id': 1,
                        'bbox': [0, 0, 128, 128], 'confidence': .9}]}) + '\n')
            writer.release()
            reads.write_text('')
            build_track_votes(observations, reads, video, roster, 'test', votes)
            item = json.loads(votes.read_text())['tracks']['0:1']
            self.assertEqual(item['counts'], {'dark': 7, 'gold': 3})
            self.assertEqual(item['team_id'], 'dark')
            self.assertEqual(item['classified_frames'], 10)

    def test_whole_track_majorities_require_clear_winner(self):
        self.assertEqual(majority(['20', '20', '20', '7'], min_votes=3)[0], '20')
        self.assertIsNone(majority(['20', '20', '7', '7'], min_votes=3)[0])
        self.assertIsNone(majority(['dark', 'dark'], min_votes=5)[0])

    def test_digit_only_reads_vote_across_track_despite_one_bad_read(self):
        roster = Roster({'schema_version': 1, 'teams': [
            {'team_id': 'a', 'name': 'A', 'players': [{'number': '20', 'name': 'Alice'},
                                                  {'number': '7', 'name': 'Other'}]},
            {'team_id': 'b', 'name': 'B', 'players': [{'number': '9', 'name': 'Bob'}]},
        ]})
        with tempfile.TemporaryDirectory() as directory:
            reads = Path(directory) / 'reads.jsonl'
            with reads.open('w') as output:
                for frame, number in enumerate(('20', '20', '7', '20')):
                    output.write(json.dumps({'decoder': 'digits_only_v3', 'segment_id': 0,
                        'track_id': 4, 'frame_index': frame * 8,
                        'raw_number': number, 'confidence': .96}) + '\n')
            self.assertEqual(track_assignments(reads, roster)[(0, 4)]['player_name'], 'Alice')

    def test_repeated_reads_merge_tracks_and_relabel_earlier_frames(self):
        roster = Roster({'schema_version': 1, 'teams': [
            {'team_id': 'a', 'name': 'A', 'players': [{'number': '23', 'name': 'Alice'}]},
            {'team_id': 'b', 'name': 'B', 'players': [{'number': '9', 'name': 'Bob'}]},
        ]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations, tracks, reads = (root / name for name in
                                            ('observations.jsonl', 'tracks.jsonl', 'jersey_reads.jsonl'))
            with observations.open('w') as output:
                for frame in range(4):
                    track = 1 if frame < 2 else 42
                    output.write(json.dumps({'frame_index': frame, 'timestamp_s': frame / 10,
                        'segment_id': 0, 'included': True, 'persons': [{
                            'role': 'player', 'track_id': track, 'position_m': [frame / 10, 0],
                            'ground_point_method': 'bbox', 'jersey': {'status': 'unknown'}}]}) + '\n')
            with reads.open('w') as output:
                for track in (1, 42):
                    for frame in (0, 1):
                        output.write(json.dumps({'segment_id': 0, 'track_id': track,
                            'frame_index': frame, 'raw_number': '23', 'confidence': .85}) + '\n')
            config = {'max_gap_s': .2, 'max_speed_m_s': 12, 'smoothing_tau_s': 0,
                      'deadband_m': 0}
            metric, count = replay_artifacts(observations, tracks, reads, roster, None, config, 'test')
            self.assertEqual(count, 2)
            with observations.open() as source:
                self.assertEqual({row['resolved_identity']['player_name'] for line in source
                                  for row in json.loads(line)['persons']}, {'Alice'})
            players = metric.result()['players']
            self.assertEqual(len(players), 1)
            self.assertEqual(players[0]['subject_id'], 'roster:a:23')
            self.assertEqual(len(players[0]['track_refs']), 2)
            self.assertAlmostEqual(players[0]['observed_distance_m'], .2)

    def test_shared_number_requires_team_color_evidence(self):
        roster = Roster({'schema_version': 1, 'teams': [
            {'team_id': 'a', 'name': 'A', 'players': [{'number': '10', 'name': 'Alice'}]},
            {'team_id': 'b', 'name': 'B', 'players': [{'number': '10', 'name': 'Bob'}]},
        ]})
        with tempfile.TemporaryDirectory() as directory:
            reads = Path(directory) / 'reads.jsonl'
            reads.write_text(''.join(json.dumps({'segment_id': 0, 'track_id': 2,
                'raw_number': '10', 'confidence': .99}) + '\n' for _ in range(2)))
            self.assertEqual(track_assignments(reads, roster), {})

    def test_final_vote_discards_stale_causal_number(self):
        roster = Roster({'schema_version': 1, 'teams': [
            {'team_id': 'dark', 'name': 'Dark', 'players': [
                {'number': '7', 'name': 'Old'}, {'number': '20', 'name': 'New'}]},
            {'team_id': 'gold', 'name': 'Gold', 'players': [
                {'number': '11', 'name': 'Other'}]},
        ]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations, tracks, reads, votes = (root / name for name in
                ('observations.jsonl', 'tracks.jsonl', 'jersey_reads.jsonl', 'team_votes.json'))
            observations.write_text(json.dumps({'frame_index': 0, 'timestamp_s': 0,
                'segment_id': 0, 'included': True, 'persons': [{
                    'role': 'player', 'track_id': 1, 'position_m': [0, 0],
                    'ground_point_method': 'bbox', 'jersey': {'status': 'confirmed',
                        'number': '7', 'team_id': 'dark', 'player_name': 'Old',
                        'identity_status': 'unique_number'}}]}) + '\n')
            reads.write_text(''.join(json.dumps({'segment_id': 0, 'track_id': 1,
                'frame_index': frame, 'raw_number': '7', 'confidence': .95}) + '\n'
                for frame in (0, 1)))
            votes.write_text(json.dumps({'evidence_version': 3,
                'roster_sha256': roster.sha256, 'tracks': {'0:1': {
                    'frames': [0], 'classified': {}, 'counts': {}, 'team_id': None,
                    'majority_share': 0, 'number_majority': None}}}))
            config = {'max_gap_s': .2, 'max_speed_m_s': 12,
                      'smoothing_tau_s': 0, 'deadband_m': 0}
            self.assertEqual(track_assignments(reads, roster,
                team_votes_path=votes), {})
            metric, count = replay_artifacts(observations, tracks, reads, roster,
                None, config, 'test', team_votes_path=votes)
            self.assertEqual(count, 0)
            person = json.loads(observations.read_text())['persons'][0]
            self.assertIsNone(person['majority_jersey_number'])
            self.assertNotIn('resolved_identity', person)
            self.assertIsNone(metric.result()['players'][0]['jersey_number'])

    def test_shared_number_uses_repeated_uniform_color(self):
        roster = Roster({'schema_version': 1, 'teams': [
            {'team_id': 'dark', 'name': 'Dark', 'uniform_color': '#202020',
             'players': [{'number': '10', 'name': 'Alice'}]},
            {'team_id': 'gold', 'name': 'Gold', 'uniform_color': '#FDB927',
             'players': [{'number': '10', 'name': 'Bob'}]},
        ]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'sample.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (128, 128))
            for _ in range(3):
                writer.write(np.full((128, 128, 3), (39, 185, 253), dtype=np.uint8))
            writer.release()
            reads = root / 'reads.jsonl'
            with reads.open('w') as output:
                for frame in (0, 1):
                    output.write(json.dumps({'segment_id': 0, 'track_id': 2,
                        'frame_index': frame, 'crop_bbox': [16, 16, 112, 112],
                        'raw_number': '10', 'confidence': .99}) + '\n')
            assignment = track_assignments(reads, roster, video)
            self.assertEqual(assignment[(0, 2)]['player_name'], 'Bob')

    def test_tracker_switch_keeps_number_on_matching_torso_only(self):
        roster = Roster({'schema_version': 1, 'teams': [
            {'team_id': 'dark', 'name': 'Dark', 'uniform_color': '#202020',
             'players': [{'number': '10', 'name': 'Alice'}]},
            {'team_id': 'gold', 'name': 'Gold', 'uniform_color': '#FDB927',
             'players': [{'number': '10', 'name': 'Bob'}]},
        ]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, reads, observations, tracks = (root / name for name in
                ('sample.avi', 'reads.jsonl', 'observations.jsonl', 'tracks.jsonl'))
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (128, 128))
            for frame in range(6):
                color = (39, 185, 253) if frame == 3 else (32, 32, 32)
                writer.write(np.full((128, 128, 3), color, dtype=np.uint8))
            writer.release()
            with reads.open('w') as output:
                for frame in (0, 1):
                    output.write(json.dumps({'segment_id': 0, 'track_id': 1, 'frame_index': frame,
                        'crop_bbox': [16, 16, 112, 112], 'raw_number': '10', 'confidence': .99}) + '\n')
            with observations.open('w') as output:
                for frame in range(6):
                    output.write(json.dumps({'frame_index': frame, 'timestamp_s': frame / 10,
                        'segment_id': 0, 'included': True, 'persons': [{
                            'role': 'player', 'track_id': 1, 'bbox': [0, 0, 128, 128],
                            'position_m': [frame / 10, 0], 'ground_point_method': 'bbox',
                            'jersey': {'number': '10', 'status': 'confirmed'}}]}) + '\n')
            config = {'max_gap_s': .2, 'max_speed_m_s': 12, 'smoothing_tau_s': 0,
                      'deadband_m': 0}
            replay_artifacts(observations, tracks, reads, roster, video, config, 'test')
            with observations.open() as source:
                people = [json.loads(line)['persons'][0] for line in source]
            self.assertEqual(people[0]['resolved_identity']['player_name'], 'Alice')
            self.assertTrue(people[3]['jersey_number_suppressed'])
            self.assertNotIn('resolved_identity', people[3])
            self.assertEqual(people[5]['resolved_identity']['player_name'], 'Alice')


if __name__ == '__main__':
    unittest.main()
