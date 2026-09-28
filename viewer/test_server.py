"""Integration checks for the viewer's artifact and video endpoints."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from viewer.server import Handler, ViewerServer


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
        (run / 'run.json').write_text(json.dumps({
            'run_id': 'test', 'status': 'completed', 'frames': 3,
            'source': {'path': str(video), 'nominal_fps': 10, 'width': 100, 'height': 50},
        }))
        (run / 'statistics.json').write_text(json.dumps({'players': []}))
        with (run / 'observations.jsonl').open('w') as file:
            for i in range(3):
                file.write(json.dumps({'frame_index': i, 'timestamp_s': i / 10, 'persons': [],
                                       'width': 100, 'height': 50}) + '\n')
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
        self.assertEqual(json.loads(body)['timestamps'], [0, .1, .2])
        _, _, body = self.get('/api/frame?name=match&index=0')
        self.assertIsNone(json.loads(body)['total_distance_m'])
        self.assertIsNone(json.loads(body)['metrics'][0]['distance_m'])
        _, _, body = self.get('/api/frame?name=match&index=2')
        self.assertEqual(json.loads(body)['total_distance_m'], 1.25)
        self.assertEqual(json.loads(body)['metrics'][0]['measured_duration_s'], .1)

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


if __name__ == '__main__':
    unittest.main()
