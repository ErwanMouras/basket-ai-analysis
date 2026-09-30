import json
import tempfile
import unittest
from pathlib import Path

from pipeline.identity import replay_artifacts
from pipeline.spatial_identity import (_neighbor_team, _track_spans,
                                       expand_direct_identity_spans, find_spatial_handoffs)
from training.jersey.roster import Roster


PLAYER = {'team_id': 'home', 'number': '23', 'player_name': 'Alice'}
DISTANCE_CONFIG = {'max_gap_s': .2, 'max_speed_m_s': 12,
                   'smoothing_tau_s': 0, 'deadband_m': 0}


def person(track, x, *, point=None):
    return {'role': 'player', 'track_id': track, 'bbox': [x - 10, 0, x + 10, 100],
            'position_m': point, 'ground_point_method': 'bbox',
            'jersey': {'status': 'unknown'}}


class SpatialIdentityTests(unittest.TestCase):
    def prepare_reverse(self, root, *, blockers=None, scene_cut=False):
        observations = root / 'observations.jsonl'
        reads = root / 'jersey_reads.jsonl'
        with observations.open('w') as output:
            for frame in range(8):
                people = [person(1, 100, point=[3, 4])] if frame < 4 else [
                    person(2, 105, point=[3.1, 4])]
                if blockers and frame in blockers:
                    people.append(person(3, blockers[frame]))
                output.write(json.dumps({'frame_index': frame,
                    'timestamp_s': frame / 30, 'segment_id': 0,
                    'included': True, 'scene_cut': scene_cut and frame == 4,
                    'persons': people}) + '\n')
        reads.write_text('')
        votes = {
            (0, 1): {'team_id': 'home', 'counts': {'home': 4},
                     'raw': {frame: 'home' for frame in range(4)}},
            (0, 2): {'team_id': 'home', 'counts': {'home': 4},
                     'raw': {frame: 'home' for frame in range(4, 8)}},
        }
        return observations, reads, votes

    def test_reverse_handoff_relabels_prior_track_and_replays_distance(self):
        roster = Roster({'schema_version': 1, 'teams': [
            {'team_id': 'home', 'name': 'Home', 'players': [{'number': '23', 'name': 'Alice'}]},
            {'team_id': 'away', 'name': 'Away', 'players': [{'number': '11', 'name': 'Bob'}]},
        ]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations, reads, votes = self.prepare_reverse(root)
            handoffs = find_spatial_handoffs(observations, reads,
                                             {(0, 2): PLAYER}, votes)
            self.assertEqual(list(handoffs), [(0, 1, 0)])
            self.assertEqual(handoffs[(0, 1, 0)]['direction'], 'backward')
            self.assertEqual(handoffs[(0, 1, 0)]['source_start_frame'], 4)
            reads.write_text(''.join(json.dumps({'segment_id': 0, 'track_id': 2,
                'frame_index': frame, 'raw_number': '23', 'confidence': .95}) + '\n'
                for frame in (4, 6)))
            metric, count = replay_artifacts(observations, root / 'tracks.jsonl',
                reads, roster, None, DISTANCE_CONFIG, 'test')
            self.assertEqual(count, 2)
            self.assertEqual(metric.spatial_handoffs[0]['direction'], 'backward')
            rows = [json.loads(line) for line in observations.read_text().splitlines()]
            self.assertEqual(rows[0]['persons'][0]['resolved_identity']['player_name'], 'Alice')
            self.assertEqual(rows[0]['persons'][0]['spatial_handoff']['source_start_frame'], 4)
            self.assertEqual(len(metric.result()['players'][0]['track_refs']), 2)

    def test_reverse_handoff_rejects_screen_cut_color_or_number_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations, reads, votes = self.prepare_reverse(root, blockers={3: 145})
            self.assertEqual(find_spatial_handoffs(observations, reads,
                                                   {(0, 2): PLAYER}, votes), {})
            observations, reads, votes = self.prepare_reverse(root, scene_cut=True)
            self.assertEqual(find_spatial_handoffs(observations, reads,
                                                   {(0, 2): PLAYER}, votes), {})
            observations, reads, votes = self.prepare_reverse(root)
            votes[(0, 1)]['raw'][3] = 'away'
            self.assertEqual(find_spatial_handoffs(observations, reads,
                                                   {(0, 2): PLAYER}, votes), {})
            votes[(0, 1)]['raw'][3] = 'home'
            reads.write_text(json.dumps({'decoder': 'digits_only_v3',
                'segment_id': 0, 'track_id': 1, 'frame_index': 2,
                'number': '11', 'confidence': .95}) + '\n')
            self.assertEqual(find_spatial_handoffs(observations, reads,
                                                   {(0, 2): PLAYER}, votes), {})

    def test_reverse_handoff_rejects_tracker_id_still_visible_at_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            observations, reads, votes = self.prepare_reverse(Path(directory))
            rows = [json.loads(line) for line in observations.read_text().splitlines()]
            for frame in range(4, 8):
                rows[frame]['persons'].append(person(1, 105))
                votes[(0, 1)]['raw'][frame] = 'away'
            votes[(0, 1)]['team_id'] = 'away'
            votes[(0, 1)]['counts'] = {'home': 4, 'away': 4}
            observations.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(find_spatial_handoffs(observations, reads,
                                                   {(0, 2): PLAYER}, votes), {})

    def test_reverse_handoff_rejects_two_possible_later_players(self):
        with tempfile.TemporaryDirectory() as directory:
            observations, reads, votes = self.prepare_reverse(Path(directory))
            rows = [json.loads(line) for line in observations.read_text().splitlines()]
            for frame in range(4, 8):
                rows[frame]['persons'][0] = person(2, 65)
                rows[frame]['persons'].append(person(3, 135))
            votes[(0, 3)] = {'team_id': 'home', 'counts': {'home': 4},
                             'raw': {frame: 'home' for frame in range(4, 8)}}
            observations.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            later = {(0, 2): PLAYER,
                     (0, 3): {'team_id': 'home', 'number': '11', 'player_name': 'Bob'}}
            self.assertEqual(find_spatial_handoffs(observations, reads, later, votes), {})

    def prepare(self, root, *, blockers=None, target_color='home', extra_reads=()):
        observations = root / 'observations.jsonl'
        reads = root / 'jersey_reads.jsonl'
        with observations.open('w') as output:
            for frame in range(8):
                people = [person(1, 100, point=[3, 4])] if frame < 4 else [
                    person(2, 105, point=[3.1, 4])]
                if blockers and frame in blockers:
                    people.append(person(3, blockers[frame], point=[3.5, 4]))
                output.write(json.dumps({'frame_index': frame, 'timestamp_s': frame / 10,
                    'segment_id': 0, 'included': True, 'scene_cut': False,
                    'persons': people}) + '\n')
        reads.write_text(''.join(json.dumps(read) + '\n' for read in extra_reads))
        votes = {(0, 2): {'team_id': target_color,
                'raw': {frame: target_color for frame in range(4, 8)}}}
        return observations, reads, votes

    def test_clear_handoff_merges_player_and_replays_distance(self):
        roster = Roster({'schema_version': 1, 'teams': [
            {'team_id': 'home', 'name': 'Home', 'players': [{'number': '23', 'name': 'Alice'}]},
            {'team_id': 'away', 'name': 'Away', 'players': [{'number': '11', 'name': 'Bob'}]},
        ]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations, reads, _ = self.prepare(root, extra_reads=[
                {'segment_id': 0, 'track_id': 1, 'frame_index': frame,
                 'raw_number': '23', 'confidence': .95} for frame in (0, 2)])
            tracks = root / 'tracks.jsonl'
            metric, count = replay_artifacts(observations, tracks, reads, roster,
                                            None, DISTANCE_CONFIG, 'test')
            self.assertEqual(count, 2)
            self.assertEqual(len(metric.spatial_handoffs), 1)
            rows = [json.loads(line) for line in observations.read_text().splitlines()]
            self.assertEqual(rows[4]['persons'][0]['resolved_identity']['player_name'], 'Alice')
            self.assertEqual(rows[4]['persons'][0]['spatial_handoff']['source_track_id'], 1)
            self.assertEqual(len(metric.result()['players']), 1)
            self.assertEqual(len(metric.result()['players'][0]['track_refs']), 2)

    def test_nearby_player_blocks_screen_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            observations, reads, votes = self.prepare(Path(directory), blockers={3: 145})
            result = find_spatial_handoffs(observations, reads, {(0, 1): PLAYER}, votes)
            self.assertEqual(result, {})

    def test_duplicate_untracked_box_does_not_block_strong_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            observations, reads, votes = self.prepare(Path(directory))
            rows = [json.loads(line) for line in observations.read_text().splitlines()]
            rows[3]['persons'].append(person(None, 100))
            observations.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            result = find_spatial_handoffs(observations, reads, {(0, 1): PLAYER}, votes)
            self.assertIn((0, 2, 4), result)

    def test_short_opponent_span_uses_strong_global_color_evidence(self):
        row = {'frame_index': 2, 'segment_id': 0,
               'persons': [person(3, 120)]}
        rows = {frame: {**row, 'frame_index': frame, 'timestamp_s': frame / 30}
                for frame in (1, 2)}
        span = _track_spans(rows)[0]
        votes = {(0, 3): {'team_id': 'away', 'counts': {'away': 19, 'home': 1},
                          'raw': {1: 'away', 2: 'away'}}}
        self.assertEqual(_neighbor_team(row['persons'][0], row,
                         {(2, 0, 3): span}, votes), 'away')

    def test_player_crossing_the_short_gap_blocks_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations, reads, _ = self.prepare(root)
            with observations.open('w') as output:
                for frame in range(10):
                    people = ([person(1, 100)] if frame < 4 else
                              [person(3, 102)] if frame == 5 else
                              [person(2, 105)] if frame >= 6 else [])
                    output.write(json.dumps({'frame_index': frame,
                        'timestamp_s': frame / 30, 'segment_id': 0,
                        'included': True, 'scene_cut': False,
                        'persons': people}) + '\n')
            votes = {(0, 2): {'team_id': 'home',
                              'raw': {frame: 'home' for frame in range(6, 10)}}}
            self.assertEqual(find_spatial_handoffs(observations, reads,
                                                   {(0, 1): PLAYER}, votes), {})
            rows = [json.loads(line) for line in observations.read_text().splitlines()]
            rows[5]['persons'] = []
            observations.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertIn((0, 2, 6), find_spatial_handoffs(observations, reads,
                                                           {(0, 1): PLAYER}, votes))

    def test_later_team_or_number_conflict_blocks_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            observations, reads, votes = self.prepare(Path(directory))
            votes[(0, 2)]['raw'][7] = 'away'
            self.assertEqual(find_spatial_handoffs(observations, reads,
                                                   {(0, 1): PLAYER}, votes), {})
            votes[(0, 2)]['raw'][7] = 'home'
            reads.write_text(json.dumps({'decoder': 'digits_only_v3', 'segment_id': 0,
                'track_id': 2, 'frame_index': 7, 'number': '11', 'confidence': .95}) + '\n')
            self.assertEqual(find_spatial_handoffs(observations, reads,
                                                   {(0, 1): PLAYER}, votes), {})

    def test_duplicate_direct_identity_at_source_endpoint_blocks_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            observations, reads, votes = self.prepare(Path(directory), blockers={3: 500})
            self.assertEqual(find_spatial_handoffs(observations, reads,
                {(0, 1): PLAYER, (0, 3): PLAYER}, votes), {})

    def test_image_handoff_handles_a_reused_id_and_opposing_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations, reads, _ = self.prepare(root)
            with observations.open('w') as output:
                for frame in range(12):
                    people = [person(3, 120, point=[2, 0])]
                    if frame <= 3 or frame == 11:
                        people.append(person(1, 100 if frame <= 3 else 500,
                                             point=[0, 0]))
                    if frame <= 5:
                        people.append(person(2, 120, point=[2, 0]))
                    if frame >= 7:
                        people.append(person(2, 140, point=[10, 0]))
                    output.write(json.dumps({'frame_index': frame,
                        'timestamp_s': frame / 30, 'segment_id': 0,
                        'included': True, 'scene_cut': False,
                        'persons': people}) + '\n')
            votes = {
                (0, 1): {'team_id': 'home', 'counts': {'home': 4},
                         'raw': {frame: 'home' for frame in range(4)}},
                (0, 2): {'team_id': 'away', 'counts': {'away': 6, 'home': 5},
                         'raw': {**{frame: 'away' for frame in range(6)},
                                 **{frame: 'home' for frame in range(7, 12)}}},
                (0, 3): {'team_id': 'away', 'counts': {'away': 12},
                         'raw': {frame: 'away' for frame in range(12)}},
            }
            handoffs = find_spatial_handoffs(observations, reads,
                {(0, 1): {**PLAYER, '_valid_frames': {0, 1, 11}}}, votes)
            self.assertEqual(list(handoffs), [(0, 2, 7)])
            self.assertEqual(handoffs[(0, 2, 7)]['end_frame'], 10)
            self.assertGreater(handoffs[(0, 2, 7)]['distance_m'], 1.2)
            votes[(0, 3)] = {'team_id': 'home', 'counts': {'home': 12},
                             'raw': {frame: 'home' for frame in range(12)}}
            self.assertEqual(find_spatial_handoffs(observations, reads,
                {(0, 1): {**PLAYER, '_valid_frames': {0, 1, 11}}}, votes), {})

    def test_reused_track_id_is_attributed_only_during_matching_span(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations, reads, votes = self.prepare(root)
            rows = [json.loads(line) for line in observations.read_text().splitlines()]
            rows[0]['persons'].append(person(2, 500, point=[20, 4]))
            observations.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            handoffs = find_spatial_handoffs(observations, reads, {(0, 1): PLAYER}, votes)
            self.assertEqual(list(handoffs), [(0, 2, 4)])
            self.assertEqual(handoffs[(0, 2, 4)]['end_frame'], 7)

    def test_sustained_color_switch_limits_handoff_even_without_tracker_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            observations, reads, _ = self.prepare(Path(directory))
            with observations.open('w') as output:
                for frame in range(15):
                    people = [person(1, 100)] if frame < 4 else [person(2, 105)]
                    output.write(json.dumps({'frame_index': frame,
                        'timestamp_s': frame / 30, 'segment_id': 0,
                        'included': True, 'scene_cut': False,
                        'persons': people}) + '\n')
            votes = {(0, 1): {'team_id': 'home', 'counts': {'home': 4},
                              'raw': {frame: 'home' for frame in range(4)}},
                     (0, 2): {'team_id': 'away', 'counts': {'home': 6, 'away': 5},
                              'raw': {frame: 'home' if frame < 10 else 'away'
                                      for frame in range(4, 15)}}}
            assigned = {(0, 1): PLAYER,
                        (0, 2): {'team_id': 'away', 'number': '11',
                                 '_valid_frames': set(range(10, 15))}}
            handoffs = find_spatial_handoffs(observations, reads, assigned, votes)
            self.assertEqual(list(handoffs), [(0, 2, 4)])
            self.assertEqual(handoffs[(0, 2, 4)]['end_frame'], 9)

    def test_clean_contiguous_span_keeps_identity_through_missing_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            observations = Path(directory) / 'observations.jsonl'
            with observations.open('w') as output:
                for frame in range(9):
                    output.write(json.dumps({'frame_index': frame,
                        'timestamp_s': frame / 30, 'segment_id': 0,
                        'persons': [person(1, 100)] if frame != 7 else []}) + '\n')
            assigned = {(0, 1): {**PLAYER, '_valid_frames': {0, 1, 2, 3, 4}}}
            votes = {(0, 1): {'team_id': 'home', 'counts': {'home': 5, 'away': 1},
                'raw': {**{frame: 'home' for frame in range(5)}, 5: None,
                        6: None, 8: 'away'}}}
            expand_direct_identity_spans(observations, assigned, votes)
            self.assertEqual(assigned[(0, 1)]['_valid_frames'], set(range(7)))


if __name__ == '__main__':
    unittest.main()
