"""Local, read-only viewer for completed analysis runs.

Run with: python -m viewer.server
"""
from __future__ import annotations

import argparse
import bisect
import json
import mimetypes
import re
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent


@dataclass
class RunIndex:
    observation_size: int
    observation_mtime_ns: int
    offsets: list[int]
    timestamps: list[float]
    history: dict[str, tuple[list[int], list[tuple[float, float, bool]]]]


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, runs_root: Path, video_root: Path):
        super().__init__(address, handler)
        self.runs_root = runs_root.resolve()
        self.video_root = video_root.resolve()
        self.indices: dict[Path, RunIndex] = {}

    def run_path(self, name: str) -> Path:
        path = (self.runs_root / name).resolve()
        if (not name or not path.is_relative_to(self.runs_root)
                or not (path / 'run.json').is_file()
                or not (path / 'observations.jsonl').is_file()
                or not (path / 'statistics.json').is_file()):
            raise FileNotFoundError('Run introuvable')
        if json.loads((path / 'run.json').read_text(encoding='utf-8')).get('status') != 'completed':
            raise FileNotFoundError('Le run n’est pas terminé')
        return path

    def index(self, path: Path) -> RunIndex:
        source = path / 'observations.jsonl'
        stat = source.stat()
        cached = self.indices.get(path)
        if cached and cached.observation_size == stat.st_size and cached.observation_mtime_ns == stat.st_mtime_ns:
            return cached
        offsets: list[int] = []
        timestamps: list[float] = []
        with source.open('rb') as file:
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                row = json.loads(line)
                offsets.append(offset)
                timestamps.append(float(row['timestamp_s']))
        series: dict[str, list[tuple[int, float | None, float]]] = {}
        tracks = path / 'tracks.jsonl'
        if tracks.is_file():
            with tracks.open(encoding='utf-8') as file:
                for line in file:
                    row = json.loads(line)
                    subject = row.get('subject_id')
                    if subject is None:
                        continue
                    delta = row.get('distance_m')
                    series.setdefault(subject, []).append((int(row['frame_index']),
                        float(delta) if delta is not None else None,
                        float(row.get('measured_duration_s') or 0)))
        history = {}
        for subject, values in series.items():
            frames, cumulative = [], []
            distance = duration = 0.0
            measured = False
            for frame, delta, seconds in values:
                if delta is not None:
                    distance += delta
                    measured = True
                duration += seconds
                frames.append(frame)
                cumulative.append((distance, duration, measured))
            history[subject] = (frames, cumulative)
        result = RunIndex(stat.st_size, stat.st_mtime_ns, offsets, timestamps, history)
        self.indices[path] = result
        return result


class Handler(BaseHTTPRequestHandler):
    server: ViewerServer

    def do_GET(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == '/api/runs':
                runs = []
                for manifest in self.server.runs_root.rglob('run.json'):
                    try:
                        data = json.loads(manifest.read_text(encoding='utf-8'))
                        if (data.get('status') != 'completed'
                                or not (manifest.parent / 'observations.jsonl').is_file()
                                or not (manifest.parent / 'statistics.json').is_file()):
                            continue
                        name = str(manifest.parent.relative_to(self.server.runs_root))
                        source = data.get('source', {})
                        runs.append({'name': name, 'video': Path(source.get('path', '')).stem,
                                     'frames': data.get('frames', 0), 'fps': source.get('nominal_fps'),
                                     'roster': (manifest.parent / 'roster.json').is_file()})
                    except (OSError, ValueError, KeyError):
                        continue
                runs.sort(key=lambda x: x['name'])
                return self.json({'runs': runs})
            if parsed.path in ('/api/run', '/api/frame', '/api/video'):
                name = query.get('name', [''])[0]
                path = self.server.run_path(name)
                manifest = json.loads((path / 'run.json').read_text(encoding='utf-8'))
                source = Path(manifest['source']['path']).resolve()
                if parsed.path == '/api/video':
                    if not source.is_relative_to(self.server.video_root) or not source.is_file():
                        raise FileNotFoundError('Vidéo source inaccessible dans le dossier autorisé')
                    return self.video(source)
                if parsed.path == '/api/run':
                    index = self.server.index(path)
                    stats_path = path / 'statistics.json'
                    stats = json.loads(stats_path.read_text(encoding='utf-8')) if stats_path.is_file() else {}
                    roster_path = path / 'roster.json'
                    roster = json.loads(roster_path.read_text(encoding='utf-8')) if roster_path.is_file() else None
                    return self.json({'name': name, 'run': {
                        'run_id': manifest.get('run_id'), 'status': manifest.get('status'),
                        'source': manifest.get('source'), 'frames': manifest.get('frames'),
                        'capabilities': manifest.get('capabilities'), 'court_coordinate_system': manifest.get('court_coordinate_system')},
                        'timestamps': index.timestamps, 'statistics': stats, 'roster': roster,
                        'video_available': source.is_relative_to(self.server.video_root) and source.is_file()})
                index = self.server.index(path)
                try:
                    frame = int(query.get('index', ['0'])[0])
                except ValueError:
                    return self.error(HTTPStatus.BAD_REQUEST, 'Indice de frame invalide')
                if frame < 0 or frame >= len(index.offsets):
                    return self.error(HTTPStatus.NOT_FOUND, 'Frame introuvable')
                with (path / 'observations.jsonl').open('rb') as file:
                    file.seek(index.offsets[frame])
                    observation = json.loads(file.readline())
                metric_rows = []
                for subject, (frames, values) in index.history.items():
                    pos = bisect.bisect_right(frames, frame) - 1
                    if pos >= 0:
                        metric_rows.append({'subject_id': subject, 'distance_m': values[pos][0] if values[pos][2] else None,
                                            'measured_duration_s': values[pos][1]})
                metric_rows.sort(key=lambda x: (x['distance_m'] is not None, x['distance_m'] or 0), reverse=True)
                return self.json({'observation': observation, 'metrics': metric_rows,
                                  'total_distance_m': sum(row['distance_m'] or 0 for row in metric_rows)
                                  if any(row['distance_m'] is not None for row in metric_rows) else None})
            if parsed.path == '/':
                return self.static('index.html')
            if re.fullmatch(r'/[a-zA-Z0-9_.-]+', parsed.path):
                return self.static(parsed.path[1:])
            return self.error(HTTPStatus.NOT_FOUND, 'Ressource introuvable')
        except (FileNotFoundError, IsADirectoryError) as exc:
            return self.error(HTTPStatus.NOT_FOUND, str(exc))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return self.error(HTTPStatus.BAD_REQUEST, str(exc))

    def json(self, value):
        data = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def static(self, name):
        path = (STATIC_ROOT / name).resolve()
        if not path.is_relative_to(STATIC_ROOT) or not path.is_file():
            raise FileNotFoundError('Fichier introuvable')
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def video(self, path: Path):
        size = path.stat().st_size
        start, end = 0, size - 1
        header = self.headers.get('Range')
        if header:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
            if not match or (not match[1] and not match[2]):
                return self.range_error(size)
            if match[1]:
                start = int(match[1])
                end = min(int(match[2]), size - 1) if match[2] else size - 1
            else:
                start = max(0, size - int(match[2]))
            if start >= size or end < start:
                return self.range_error(size)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if header else HTTPStatus.OK)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if header:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as file:
            file.seek(start)
            remaining = end - start + 1
            while remaining:
                block = file.read(min(1024 * 1024, remaining))
                if not block:
                    break
                try:
                    self.wfile.write(block)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(block)

    def range_error(self, size):
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header('Content-Range', f'bytes */{size}')
        self.end_headers()

    def error(self, status, message):
        data = json.dumps({'error': message}, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description='Visualiseur local des runs vidéo')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--runs-root', type=Path, default=PROJECT_ROOT / 'runs' / 'analysis')
    parser.add_argument('--video-root', type=Path, default=PROJECT_ROOT / 'datas')
    args = parser.parse_args()
    server = ViewerServer((args.host, args.port), Handler, args.runs_root, args.video_root)
    print(f'Visualiseur : http://{args.host}:{args.port}')
    print(f'Runs : {server.runs_root}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
