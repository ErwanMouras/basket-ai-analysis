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

from training.jersey.display import display_colors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent


@dataclass
class RunIndex:
    observation_size: int
    observation_mtime_ns: int
    offsets: list[int]
    timestamps: list[float]
    history: dict[str, tuple[list[int], list[tuple[float, float, bool]]]]
    markers: list[dict]
    reads: list[dict]
    observed_jerseys: dict[str, dict]


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
        markers: list[dict] = []
        jerseys: dict[str, dict] = {}
        previous_court = None
        with source.open('rb') as file:
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                row = json.loads(line)
                offsets.append(offset)
                timestamps.append(float(row['timestamp_s']))
                if row.get('scene_cut') and row['frame_index']:
                    markers.append({'index': row['frame_index'], 'time': row['timestamp_s'], 'type': 'cut', 'label': 'Changement de plan'})
                court = (row.get('court') or {}).get('status')
                if court == 'unavailable' and previous_court != 'unavailable':
                    markers.append({'index': row['frame_index'], 'time': row['timestamp_s'], 'type': 'calibration', 'label': 'Calibration indisponible'})
                previous_court = court
                for person in row.get('persons', []):
                    if person.get('role') != 'player' or person.get('track_id') is None:
                        continue
                    jersey = person.get('jersey') or {}
                    key = f"{row.get('segment_id', 0)}:{person['track_id']}"
                    item = jerseys.setdefault(key, {'numbers': set(), 'confirmed_frames': 0, 'groups': set()})
                    if 'majority_jersey_number' in person:
                        number = person['majority_jersey_number']
                    else:
                        number = jersey.get('number') if jersey.get('status') == 'confirmed' else None
                    if not person.get('jersey_number_suppressed') and number is not None:
                        item['numbers'].add(number)
                        item['confirmed_frames'] += 1
                    if 'majority_jersey_number' not in person and jersey.get('team_group'):
                        item['groups'].add(jersey['team_group'])
        reads = []
        read_path = path / 'jersey_reads.jsonl'
        final_votes = (path / 'team_votes.json').is_file()
        if read_path.is_file():
            with read_path.open(encoding='utf-8') as file:
                for line in file:
                    read = json.loads(line)
                    reads.append(read)
                    if read.get('status') == 'vote' and (not final_votes or read.get('decoder') in ('digits_only_v3', 'digits_only_v4')):
                        frame = read.get('frame_index')
                        if isinstance(frame, int) and 0 <= frame < len(timestamps):
                            markers.append({'index': frame, 'time': timestamps[frame], 'type': 'jersey', 'label': 'Lecture de maillot retenue'})
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
        markers.sort(key=lambda marker: (marker['index'], marker['type']))
        observed_jerseys = {key: {'number': next(iter(item['numbers'])) if len(item['numbers']) == 1 else None,
                                  'confirmed_frames': item['confirmed_frames'],
                                  'conflicting_numbers': sorted(item['numbers']) if len(item['numbers']) > 1 else [],
                                  'team_group': next(iter(item['groups'])) if len(item['groups']) == 1 else None}
                            for key, item in jerseys.items()}
        result = RunIndex(stat.st_size, stat.st_mtime_ns, offsets, timestamps, history, markers, reads, observed_jerseys)
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
                        roster_path = manifest.parent / 'roster.json'
                        roster = json.loads(roster_path.read_text(encoding='utf-8')) if roster_path.is_file() else None
                        match = (roster or {}).get('match') or {}
                        video_path = Path(source.get('path', ''))
                        source_match = f'{video_path.parent.parent.name}/{video_path.parent.name}' if video_path.name == 'video.mp4' else video_path.stem
                        fps = source.get('nominal_fps') or 0
                        frames = data.get('frames', 0)
                        runs.append({'name': name, 'video': video_path.stem,
                                     'frames': frames, 'fps': fps, 'duration_s': (frames - 1) / fps if fps and frames else 0,
                                     'match': match.get('match_id') or source_match, 'date': match.get('date'),
                                     'roster': roster is not None,
                                     'video_available': Path(source.get('path', '')).resolve().is_relative_to(self.server.video_root) and Path(source.get('path', '')).is_file()})
                    except (OSError, ValueError, KeyError):
                        continue
                runs.sort(key=lambda x: x['name'])
                return self.json({'runs': runs})
            if parsed.path in ('/api/run', '/api/frame', '/api/frames', '/api/reads', '/api/video'):
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
                    vote_path = path / 'team_votes.json'
                    vote_summaries = None
                    if vote_path.is_file():
                        vote_data = json.loads(vote_path.read_text(encoding='utf-8'))
                        vote_summaries = {key: {
                            'number_attempts': item.get('number_attempts', 0),
                            'number_votes': item.get('number_votes', {}),
                            'number_majority': item.get('number_majority'),
                            'number_majority_share': item.get('number_majority_share'),
                            'team_id': item.get('team_id'),
                            'team_votes': item.get('counts', {}),
                            'classified_frames': item.get('classified_frames', 0),
                            'visible_frames': len(item.get('frames', [])),
                        } for key, item in vote_data.get('tracks', {}).items()}
                    reference_roster = None
                    if roster is None and source.is_relative_to(self.server.video_root):
                        reference_path = (source.parent / 'jersey.json').resolve()
                        if (reference_path.is_relative_to(self.server.video_root) and reference_path.is_file()
                                and reference_path.stat().st_size <= 1024 * 1024):
                            try:
                                candidate = json.loads(reference_path.read_text(encoding='utf-8'))
                                if (isinstance(candidate, dict) and candidate.get('schema_version') == 1
                                        and isinstance(candidate.get('teams'), list) and len(candidate['teams']) == 2):
                                    reference_roster = candidate
                            except (OSError, ValueError):
                                pass
                    palette = manifest.get('display_colors') or display_colors(roster)
                    return self.json({'name': name, 'run': {
                        'run_id': manifest.get('run_id'), 'status': manifest.get('status'),
                        'source': manifest.get('source'), 'frames': manifest.get('frames'),
                        'capabilities': manifest.get('capabilities'), 'court_coordinate_system': manifest.get('court_coordinate_system'),
                        'display_colors': palette, 'identity_resolution': manifest.get('identity_resolution')},
                        'timestamps': index.timestamps, 'statistics': stats, 'roster': roster,
                        'reference_roster': reference_roster,
                        'reference_display_colors': display_colors(reference_roster) if reference_roster else None,
                        'observed_jerseys': index.observed_jerseys,
                        'track_votes': vote_summaries,
                        'spatial_handoffs': manifest.get('spatial_handoffs', []),
                        'markers': index.markers,
                        'video_available': source.is_relative_to(self.server.video_root) and source.is_file()})
                index = self.server.index(path)
                if parsed.path == '/api/reads':
                    track = query.get('track', [''])[0]
                    segment = query.get('segment', [''])[0]
                    if not track.isdecimal() or not segment.isdecimal():
                        return self.error(HTTPStatus.BAD_REQUEST, 'Track ou segment invalide')
                    selected_reads = [row for row in index.reads
                                      if row.get('track_id') == int(track) and row.get('segment_id') == int(segment)]
                    modern_reads = [row for row in selected_reads if row.get('decoder') == 'digits_only_v4']
                    if not modern_reads:
                        modern_reads = [row for row in selected_reads if row.get('decoder') == 'digits_only_v3']
                    return self.json({'reads': modern_reads if (path / 'team_votes.json').is_file()
                                      else selected_reads})
                if parsed.path == '/api/frames':
                    try:
                        start = int(query.get('start', ['0'])[0])
                        count = int(query.get('count', ['1'])[0])
                    except ValueError:
                        return self.error(HTTPStatus.BAD_REQUEST, 'Plage de frames invalide')
                    if start < 0 or not 1 <= count <= 60 or start >= len(index.offsets):
                        return self.error(HTTPStatus.BAD_REQUEST, 'Plage de frames invalide')
                    return self.json({'frames': [self.frame_payload(path, index, i)
                                                 for i in range(start, min(start + count, len(index.offsets)))]})
                try:
                    frame = int(query.get('index', ['0'])[0])
                except ValueError:
                    return self.error(HTTPStatus.BAD_REQUEST, 'Indice de frame invalide')
                if frame < 0 or frame >= len(index.offsets):
                    return self.error(HTTPStatus.NOT_FOUND, 'Frame introuvable')
                return self.json(self.frame_payload(path, index, frame))
            if parsed.path == '/':
                return self.static('index.html')
            if re.fullmatch(r'/[a-zA-Z0-9_.-]+', parsed.path):
                return self.static(parsed.path[1:])
            return self.error(HTTPStatus.NOT_FOUND, 'Ressource introuvable')
        except (FileNotFoundError, IsADirectoryError) as exc:
            return self.error(HTTPStatus.NOT_FOUND, str(exc))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return self.error(HTTPStatus.BAD_REQUEST, str(exc))

    def frame_payload(self, path, index, frame):
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
        return {'observation': observation, 'metrics': metric_rows,
                'total_distance_m': sum(row['distance_m'] or 0 for row in metric_rows)
                if any(row['distance_m'] is not None for row in metric_rows) else None}

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
        self.send_header('Cache-Control', 'no-store')
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
