"""Systematic, whole-track evidence for jersey numbers and uniform colors."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import cv2
import numpy as np

from training.common.files import write_json
from training.jersey.colors import color_distance, dominant_color


EVIDENCE_VERSION = 5
NUMBER_FRAME_STEP = 2
MIN_OCR_CONFIDENCE = 0.8


def uniform_signatures(roster):
    signatures = {}
    for team in roster.payload['teams']:
        color = team.get('uniform_color')
        if color:
            rgb = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
            signatures[team['team_id']] = dominant_color(
                np.tile(rgb, (32, 128, 1)).astype('uint8'))
    return signatures


def torso_box(person):
    pose = person.get('pose') or {}
    valid = pose.get('valid') or []
    if len(valid) >= 13 and pose.get('keypoints') and all(valid[i] for i in (5, 6, 11, 12)):
        points = [pose['keypoints'][i] for i in (5, 6, 11, 12)]
        left, top = min(point[0] for point in points), min(point[1] for point in points)
        right, bottom = max(point[0] for point in points), max(point[1] for point in points)
        width, height = right - left, bottom - top
        return [round(left + .10 * width), round(top + .12 * height),
                round(right - .10 * width), round(bottom - .08 * height)]
    x1, y1, x2, y2 = person['bbox']
    width, height = x2 - x1, y2 - y1
    return [round(x1 + .22 * width), round(y1 + .20 * height),
            round(x2 - .22 * width), round(y1 + .56 * height)]


def sample_team(frame, bbox, signatures):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    signature = dominant_color(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    matches = [team for team, uniform in signatures.items()
               if signature and uniform and color_distance(signature, uniform) <= 6]
    return matches[0] if len(matches) == 1 else None


def majority(votes, *, min_votes=3, min_share=0.6):
    counts = Counter(votes)
    total = sum(counts.values())
    if total < min_votes:
        return None, 0.0
    winner, count = counts.most_common(1)[0]
    share = count / total
    return (winner if count > total / 2 and share >= min_share else None), share


def split_color_episodes(frames, colors):
    """Split a tracker ID at gaps and sustained switches of uniform color."""
    if not frames:
        return []
    episodes = []
    start = 0
    current = next((colors.get(frame) for frame in frames if colors.get(frame)), None)
    for index in range(1, len(frames)):
        frame = frames[index]
        if frame != frames[index - 1] + 1:
            episodes.append(frames[start:index])
            start = index
            current = next((colors.get(later) for later in frames[index:]
                            if colors.get(later)), None)
            continue
        color = colors.get(frame)
        if color and current and color != current:
            support = [colors.get(later) for later in frames[index:min(len(frames), index + 5)]
                       if colors.get(later)]
            if support.count(color) >= 3 and support.count(current) == 0:
                episodes.append(frames[start:index])
                start = index
                current = color
        elif color and current is None:
            current = color
    episodes.append(frames[start:])
    return episodes


def _selected_frames(observations_path):
    """Candidate OCR frames before color episodes are known."""
    by_track = defaultdict(list)
    with Path(observations_path).open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            for person in row['persons']:
                if person.get('role') == 'player' and person.get('track_id') is not None:
                    by_track[(row['segment_id'], person['track_id'])].append(
                        (row['frame_index'], row['timestamp_s']))
    selected = set()
    for key, points in by_track.items():
        for episode in split_color_episodes([frame for frame, _ in points], {}):
            selected.update((key[0], key[1], frame) for frame in episode
                            if (frame - episode[0]) % NUMBER_FRAME_STEP == 0)
    return selected


def build_track_votes(observations_path, reads_path, video_path, roster, run_id,
                      output_path, jersey_config=None):
    """Classify every torso; OCR each episode every two frames until confirmed."""
    from training.jersey.config import settings
    from training.jersey.crops import candidate
    from training.jersey.reader import ParseqReader

    config = settings({**(jersey_config or {}), 'enabled': True, 'device': 'cpu'})
    signatures = uniform_signatures(roster)
    votes = defaultdict(lambda: {'frames': [], 'classified': {}, 'counts': Counter()})
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    rows = []
    try:
        with Path(observations_path).open(encoding='utf-8') as source:
            for line in source:
                row = json.loads(line)
                rows.append(row)
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"Video ended before observation frame {row['frame_index']}")
                for person in row['persons']:
                    if person.get('role') != 'player' or person.get('track_id') is None:
                        continue
                    key = (row['segment_id'], person['track_id'])
                    item = votes[key]
                    frame = row['frame_index']
                    item['frames'].append(frame)
                    team = sample_team(image, torso_box(person), signatures)
                    if team:
                        item['classified'][str(frame)] = team
                        item['counts'][team] += 1
    finally:
        cap.release()

    episodes = {}
    episode_at = {}
    for key, item in votes.items():
        colors = {frame: item['classified'].get(str(frame)) for frame in item['frames']}
        episodes[key] = []
        for frames in split_color_episodes(item['frames'], colors):
            episode = {'start_frame': frames[0], 'end_frame': frames[-1],
                       'frames': frames, 'number_attempts': 0, 'number_votes': Counter(),
                       'confirmation_frame': None}
            episodes[key].append(episode)
            for frame in frames:
                episode_at[(key, frame)] = episode

    reader = None
    rebuilt = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    try:
        for row in rows:
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Video ended before observation frame {row['frame_index']}")
            players = [person for person in row['persons']
                       if person.get('role') == 'player' and person.get('track_id') is not None]
            for person in players:
                key = (row['segment_id'], person['track_id'])
                episode = episode_at[(key, row['frame_index'])]
                if (episode['confirmation_frame'] is not None or
                        (row['frame_index'] - episode['start_frame']) % NUMBER_FRAME_STEP):
                    continue
                crops = []
                for variant in ('center', 'upper', 'lower'):
                    crop = candidate(image, person, players, config, variant=variant)
                    if crop and crop['region'] in ('pose_number', 'pose_number_upper',
                                                    'pose_number_center',
                                                    'bbox_number', 'bbox_number_upper'):
                        crops.append(crop)
                if not crops:
                    continue
                if reader is None:
                    reader = ParseqReader(config)
                results = []
                for offset in range(0, len(crops), config['batch_size']):
                    results.extend(reader.read([crop['image'] for crop in
                                                crops[offset:offset + config['batch_size']]]))
                reads = []
                for crop, result in zip(crops, results, strict=True):
                    number = result['number'] if roster.allows(result['number']) else None
                    accepted = number is not None and result['confidence'] >= MIN_OCR_CONFIDENCE
                    reads.append({'schema_version': 1, 'run_id': run_id,
                        'decoder': 'digits_only_v4', 'sampling': 'every_2_frames_until_majority',
                        'episode_start_frame': episode['start_frame'],
                        'segment_id': key[0], 'track_id': key[1],
                        'frame_index': row['frame_index'], 'processed_frame': row['frame_index'],
                        'timestamp_seconds': row['timestamp_s'], 'crop_bbox': crop['bbox'],
                        'region': crop['region'], 'quality': crop['quality'],
                        'text': result['text'], 'number': number,
                        'raw_number': result['number'], 'confidence': result['confidence'],
                        'rejection_reason': None if accepted else
                            'not_in_roster' if number is None else 'low_ocr_confidence',
                        'status': 'vote' if accepted else 'rejected'})
                best = max(reads, key=lambda read: read['confidence'])
                rebuilt.append(best)
                episode['number_attempts'] += 1
                if best['status'] == 'vote':
                    episode['number_votes'][best['number']] += 1
                number, _ = majority(episode['number_votes'].elements(),
                                     min_votes=3, min_share=.75)
                if number is not None:
                    episode['confirmation_frame'] = row['frame_index']
    finally:
        cap.release()

    with Path(reads_path).open(encoding='utf-8') as source:
        previous = [json.loads(line) for line in source]
    previous = [read for read in previous
                if read.get('decoder') not in ('digits_only_v3', 'digits_only_v4')]
    number_attempts = Counter()
    number_votes = defaultdict(Counter)
    for read in rebuilt:
        key = (read['segment_id'], read['track_id'])
        number_attempts[key] += 1
        if read['number'] is not None and read['confidence'] >= MIN_OCR_CONFIDENCE:
            number_votes[key][read['number']] += 1
    temporary = Path(reads_path).with_name(Path(reads_path).name + '.votes')
    try:
        with temporary.open('w', encoding='utf-8') as output:
            for read in previous + rebuilt:
                output.write(json.dumps(read, ensure_ascii=False, separators=(',', ':')) + '\n')
        os.replace(temporary, reads_path)
    finally:
        temporary.unlink(missing_ok=True)

    tracks = {}
    for (segment, track), item in votes.items():
        team, share = majority(item['counts'].elements(), min_votes=5, min_share=.5)
        key = (segment, track)
        number, number_share = majority(number_votes[key].elements(), min_votes=3)
        episode_summaries = []
        for episode in episodes[key]:
            counts = Counter(item['classified'].get(str(frame)) for frame in episode['frames'])
            counts.pop(None, None)
            local_team, local_share = majority(counts.elements(), min_votes=3, min_share=.6)
            local_number, local_number_share = majority(episode['number_votes'].elements(),
                                                       min_votes=3, min_share=.75)
            episode_summaries.append({
                'start_frame': episode['start_frame'], 'end_frame': episode['end_frame'],
                'team_id': local_team, 'team_votes': dict(counts),
                'team_majority_share': local_share,
                'number_attempts': episode['number_attempts'],
                'number_votes': dict(episode['number_votes']),
                'number_majority': local_number,
                'number_majority_share': local_number_share,
                'confirmation_frame': episode['confirmation_frame']})
        tracks[f'{segment}:{track}'] = {
            'frames': item['frames'], 'classified': item['classified'],
            'counts': dict(item['counts']), 'team_id': team, 'majority_share': share,
            'classified_frames': sum(item['counts'].values()),
            'number_attempts': number_attempts[key], 'number_votes': dict(number_votes[key]),
            'number_majority': number, 'number_majority_share': number_share,
            'episodes': episode_summaries,
        }
    write_json(output_path, {'schema_version': 1, 'evidence_version': EVIDENCE_VERSION,
        'roster_sha256': roster.sha256, 'number_frame_step': NUMBER_FRAME_STEP,
        'number_confirmation': {'min_votes': 3, 'min_share': .75},
        'min_ocr_confidence': MIN_OCR_CONFIDENCE,
        'ocr': reader.provenance if reader is not None else None,
        'number_reads': len(rebuilt), 'tracks': tracks})
    return {'number_reads': len(rebuilt), 'tracks': len(tracks)}


def refresh_vote_summaries(output_path, reads_path):
    """Refresh auditable number totals without repeating video or model inference."""
    output_path = Path(output_path)
    document = json.loads(output_path.read_text(encoding='utf-8'))
    if document.get('evidence_version') != EVIDENCE_VERSION:
        raise ValueError('Unsupported track vote evidence version')
    if 'ocr' not in document:
        from training.jersey.config import MODEL_SHA256

        document['ocr'] = {'decoder': 'digits_only_v4', 'checkpoint_sha256': MODEL_SHA256}
    attempts, counts = Counter(), defaultdict(Counter)
    episode_attempts, episode_counts = Counter(), defaultdict(Counter)
    with Path(reads_path).open(encoding='utf-8') as source:
        for line in source:
            read = json.loads(line)
            if read.get('decoder') != 'digits_only_v4':
                continue
            key = f"{read['segment_id']}:{read['track_id']}"
            attempts[key] += 1
            episode_key = (key, read['episode_start_frame'])
            episode_attempts[episode_key] += 1
            if read.get('number') is not None and read.get('confidence', 0) >= MIN_OCR_CONFIDENCE:
                counts[key][read['number']] += 1
                episode_counts[episode_key][read['number']] += 1
    for key, track in document['tracks'].items():
        number, share = majority(counts[key].elements(), min_votes=3)
        track.update(number_attempts=attempts[key], number_votes=dict(counts[key]),
                     number_majority=number, number_majority_share=share)
        for episode in track.get('episodes', []):
            episode_key = (key, episode['start_frame'])
            number, share = majority(episode_counts[episode_key].elements(),
                                     min_votes=3, min_share=.75)
            episode.update(number_attempts=episode_attempts[episode_key],
                           number_votes=dict(episode_counts[episode_key]),
                           number_majority=number, number_majority_share=share)
    write_json(output_path, document)
    return document
