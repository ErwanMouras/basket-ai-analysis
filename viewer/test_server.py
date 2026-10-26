"""Integration checks for the viewer's artifact and video endpoints."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from viewer.server import Handler, ViewerServer
from training.jersey.display import display_colors, person_display_color


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        runs = root / 'runs'
        run = runs / 'match'
        video = root / 'videos' / 'game.mp4'
        run.mkdir(parents=True)
        video.parent.mkdir()
        video.write_bytes(b'0123456789')
        (video.parent / 'jersey.json').write_text(json.dumps({'schema_version': 1, 'teams': [
            {'team_id': 'home', 'name': 'Home', 'players': [{'number': '23', 'name': 'A'}]},
            {'team_id': 'away', 'name': 'Away', 'players': [{'number': '11', 'name': 'B'}]}]}))
        (run / 'run.json').write_text(json.dumps({
            'run_id': 'test', 'status': 'completed', 'frames': 3,
            'source': {'path': str(video), 'nominal_fps': 10, 'width': 100, 'height': 50},
        }))
        (run / 'statistics.json').write_text(json.dumps({'players': []}))
        with (run / 'observations.jsonl').open('w') as file:
            for i in range(3):
                persons = [{'role': 'player', 'track_id': 1, 'jersey': {'status': 'confirmed', 'number': '23'}}] if i == 1 else []
                file.write(json.dumps({'frame_index': i, 'segment_id': 0, 'timestamp_s': i / 10, 'persons': persons,
                                       'scene_cut': i == 2, 'court': {'status': 'unavailable' if i == 1 else 'fit'},
                                       'width': 100, 'height': 50}) + '\n')
        (run / 'jersey_reads.jsonl').write_text(json.dumps({'frame_index': 1, 'track_id': 1,
            'segment_id': 0, 'status': 'vote', 'text': '23'}) + '\n')
        with (run / 'tracks.jsonl').open('w') as file:
            for i, delta in enumerate([None, 1.25, None]):
                file.write(json.dumps({'frame_index': i, 'subject_id': 'track-1',
                                       'distance_m': delta, 'measured_duration_s': .1 if delta else 0}) + '\n')
        self.server = ViewerServer(('127.0.0.1', 0), Handler, runs, video.parent)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def get(self, path, headers=None):
        with urlopen(Request(self.url + path, headers=headers or {})) as response:
            return response.status, response.headers, response.read()

    def test_run_frames_and_progressive_metrics(self):
        _, _, body = self.get('/api/runs')
        self.assertEqual(json.loads(body)['runs'][0]['name'], 'match')
        _, _, body = self.get('/api/run?name=match')
        meta = json.loads(body)
        self.assertEqual(meta['timestamps'], [0, .1, .2])
        self.assertEqual({m['type'] for m in meta['markers']}, {'cut', 'calibration', 'jersey'})
        self.assertEqual(meta['run']['display_colors']['referee'], '#FFBD78')
        self.assertIsNone(meta['roster'])
        self.assertEqual(meta['reference_roster']['teams'][0]['team_id'], 'home')
        self.assertEqual(meta['observed_jerseys']['0:1']['number'], '23')
        self.assertEqual(meta['spatial_handoffs'], [])
        _, _, body = self.get('/api/frame?name=match&index=0')
        self.assertIsNone(json.loads(body)['total_distance_m'])
        self.assertIsNone(json.loads(body)['metrics'][0]['distance_m'])
        _, _, body = self.get('/api/frame?name=match&index=2')
        self.assertEqual(json.loads(body)['total_distance_m'], 1.25)
        self.assertEqual(json.loads(body)['metrics'][0]['measured_duration_s'], .1)
        _, _, body = self.get('/api/frames?name=match&start=0&count=3')
        self.assertEqual([f['observation']['frame_index'] for f in json.loads(body)['frames']], [0, 1, 2])
        _, _, body = self.get('/api/reads?name=match&track=1&segment=0')
        self.assertEqual(json.loads(body)['reads'][0]['text'], '23')

    def test_video_range_and_path_boundary(self):
        status, headers, body = self.get('/api/video?name=match', {'Range': 'bytes=2-5'})
        self.assertEqual(status, 206)
        self.assertEqual(headers['Content-Range'], 'bytes 2-5/10')
        self.assertEqual(body, b'2345')
        with self.assertRaises(HTTPError) as exc:
            self.get('/api/frame?name=..%2Foutside&index=0')
        self.assertEqual(exc.exception.code, 404)
        with self.assertRaises(HTTPError) as exc:
            self.get('/api/video?name=match', {'Range': 'bytes=100-'})
        self.assertEqual(exc.exception.code, 416)

    def test_portable_relative_video_paths_stay_inside_video_root(self):
        path = self.server.runs_root / 'match' / 'run.json'
        manifest = json.loads(path.read_text())
        manifest['source']['path'] = 'game.mp4'
        path.write_text(json.dumps(manifest))
        _, _, body = self.get('/api/run?name=match')
        self.assertTrue(json.loads(body)['video_available'])
        self.assertEqual(self.get('/api/video?name=match')[2], b'0123456789')
        manifest['source']['path'] = '../outside.mp4'
        path.write_text(json.dumps(manifest))
        with self.assertRaises(HTTPError):
            self.get('/api/video?name=match')

    def test_majority_vote_summary_prefers_digit_only_reads(self):
        run = self.server.runs_root / 'match'
        (run / 'team_votes.json').write_text(json.dumps({'tracks': {'0:1': {
            'frames': [0, 1, 2], 'counts': {'home': 2}, 'team_id': 'home',
            'classified_frames': 2, 'number_attempts': 3,
            'number_votes': {'23': 2}, 'number_majority': '23',
            'number_majority_share': 1.0}}}))
        with (run / 'jersey_reads.jsonl').open('a') as output:
            output.write(json.dumps({'frame_index': 2, 'track_id': 1, 'segment_id': 0,
                                      'decoder': 'digits_only_v3', 'text': '23'}) + '\n')
            output.write(json.dumps({'frame_index': 2, 'track_id': 1, 'segment_id': 0,
                                      'decoder': 'digits_only_v4', 'text': '23'}) + '\n')
        _, _, body = self.get('/api/run?name=match')
        self.assertEqual(json.loads(body)['track_votes']['0:1']['number_majority'], '23')
        _, _, body = self.get('/api/reads?name=match&track=1&segment=0')
        self.assertEqual([read.get('decoder') for read in json.loads(body)['reads']],
                         ['digits_only_v4'])

    def test_final_votes_hide_legacy_text_reads_when_no_numeric_crop(self):
        run = self.server.runs_root / 'match'
        (run / 'team_votes.json').write_text(json.dumps({'tracks': {'0:1': {
            'frames': [1], 'counts': {}, 'team_id': None,
            'classified_frames': 0, 'number_attempts': 0,
            'number_votes': {}, 'number_majority': None}}}))
        _, _, body = self.get('/api/reads?name=match&track=1&segment=0')
        self.assertEqual(json.loads(body)['reads'], [])

    def test_team_display_palette(self):
        palette = display_colors({'teams': [
            {'team_id': 'home', 'color': '#123456'}, {'team_id': 'away'}]})
        self.assertEqual(palette['teams']['home'], '#123456')
        self.assertNotEqual(palette['teams']['home'], palette['teams']['away'])
        self.assertEqual(person_display_color({'role': 'player', 'jersey': {'team_id': 'home'}}, palette), '#123456')
        self.assertEqual(person_display_color({'role': 'player', 'jersey': {'team_group': 'team_2'}}, palette), palette['groups']['team_2'])
        self.assertEqual(person_display_color({'role': 'referee'}, palette), '#FFBD78')

    def test_conflicting_confirmed_numbers_are_not_promoted(self):
        source = Path(self.temp.name) / 'runs/match/observations.jsonl'
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        rows[2]['persons'] = [{'role': 'player', 'track_id': 1,
                              'jersey': {'status': 'confirmed', 'number': '11'}}]
        source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        _, _, body = self.get('/api/run?name=match')
        observed = json.loads(body)['observed_jerseys']['0:1']
        self.assertIsNone(observed['number'])
        self.assertEqual(observed['conflicting_numbers'], ['11', '23'])

    def test_final_majority_overrides_stale_causal_number(self):
        source = Path(self.temp.name) / 'runs/match/observations.jsonl'
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        rows[1]['persons'][0]['majority_jersey_number'] = '11'
        source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        _, _, body = self.get('/api/run?name=match')
        self.assertEqual(json.loads(body)['observed_jerseys']['0:1']['number'], '11')


class HeatmapEndpointTests(unittest.TestCase):
    tearDown = ViewerTests.tearDown
    get = ViewerTests.get
    # Reuse the HTTP fixture, while keeping the analytics in an independent module.
    def setUp(self):
        ViewerTests.setUp(self)
        self.run = self.server.runs_root / 'match'
        manifest = json.loads((self.run / 'run.json').read_text())
        manifest['court_coordinate_system'] = {'league': 'NBA', 'units': 'metres', 'length': 28.6512, 'width': 15.24}
        (self.run / 'run.json').write_text(json.dumps(manifest))
        (self.run / 'roster.json').write_text(json.dumps({'teams': [
            {'team_id': 'home', 'name': 'Home', 'players': [{'number': '23', 'name': 'A'}]},
            {'team_id': 'away', 'name': 'Away', 'players': [{'number': '11', 'name': 'B'}]}]}))
        from viewer.test_heatmaps import frame, person
        self.rows = [frame(i, i / 10, [person(team='home', number='23')]) for i in range(3)]
        (self.run / 'observations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows))

    def test_heatmap_metadata_and_time_weighted_map(self):
        _, _, body = self.get('/api/heatmaps?name=match')
        meta = json.loads(body)
        self.assertTrue(meta['available'])
        self.assertEqual(len(meta['plans']), 1)
        _, _, body = self.get('/api/heatmaps?name=match&subject=roster:home:23&mode=zones')
        heatmap = json.loads(body)['maps'][0]
        self.assertAlmostEqual(heatmap['total'], .2)
        self.assertAlmostEqual(sum(heatmap['zones']), .2)
        self.assertEqual(heatmap['unit'], 's')

    def test_bad_heatmap_queries_and_run_boundary(self):
        for suffix in ('&subject=unknown', '&subject=team:home&start=nan',
                       '&subject=team:home&end=999', '&subject=team:home&mode=comparison&compare=roster:home:23'):
            with self.assertRaises(HTTPError) as error:
                self.get('/api/heatmaps?name=match'+suffix)
            self.assertEqual(error.exception.code, 400)
        with self.assertRaises(HTTPError) as error:
            self.get('/api/heatmaps?name=..%2Foutside')
        self.assertEqual(error.exception.code, 404)

    def test_heatmap_cache_invalidates_on_tracks_only_change(self):
        tracks = self.run / 'tracks.jsonl'
        def write(distance):
            tracks.write_text(json.dumps({'frame_index': 1, 'segment_id': 0, 'track_id': 1,
                'subject_id': 'roster:home:23', 'interval_start_s': 0., 'measured_duration_s': .1,
                'distance_m': distance, 'reason': None})+'\n')
        write(.1)
        url = '/api/heatmaps?name=match&subject=team:home&mode=movement'
        self.assertAlmostEqual(json.loads(self.get(url)[2])['maps'][0]['total'], .1)
        write(.25)
        self.assertAlmostEqual(json.loads(self.get(url)[2])['maps'][0]['total'], .25)

    def test_phase_query_combines_oriented_passages_across_camera_plans(self):
        from viewer.test_phases import phase_frame, ROSTER, GEOMETRY
        manifest = json.loads((self.run / 'run.json').read_text())
        manifest['court_coordinate_system'] = GEOMETRY
        (self.run / 'run.json').write_text(json.dumps(manifest))
        (self.run / 'roster.json').write_text(json.dumps(ROSTER))
        rows = [phase_frame(i, mirror=i >= 10, segment_id=int(i >= 10)) for i in range(20)]
        (self.run / 'observations.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        _, _, body = self.get('/api/heatmaps?name=match&subject=roster:a:1&phase=offense')
        data = json.loads(body)
        self.assertEqual(data['scope']['normalized_basket'], 'right')
        self.assertIsNone(data['plan'])
        self.assertAlmostEqual(data['maps'][0]['total'], 1.8)
        _, _, body = self.get('/api/heatmaps?name=match')
        self.assertEqual(len(json.loads(body)['phase_analysis']['episodes']), 2)
        for suffix in ('&phase=made_up', '&phase=offense&episode=99'):
            with self.assertRaises(HTTPError) as error:
                self.get('/api/heatmaps?name=match&subject=team:a' + suffix)
            self.assertEqual(error.exception.code, 400)


if __name__ == '__main__':
    unittest.main()
