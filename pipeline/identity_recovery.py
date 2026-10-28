"""Recover crops rejected by bbox overlap, using saved instance masks and CUDA OCR.

Writes a separate evidence directory; the original run stays untouched. Existing
named tracks are excluded, and each source frame contributes at most one vote.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import shutil
import time

from pipeline.jersey_votes import EVIDENCE_VERSION, MIN_OCR_CONFIDENCE, refresh_vote_summaries

ANCHOR_OFFSETS = (-16, -8, -4, -2, 2, 4, 8, 16)


def sample_frames(frames, limit):
    if len(frames) <= limit:
        return set(frames)
    return {frames[round(index * (len(frames) - 1) / (limit - 1))]
            for index in range(limit)}


def merge_reads(previous, recovered):
    """Replace modern reads only when a new crop has higher confidence."""
    best = {}
    for read in recovered:
        key = (read['segment_id'], read['track_id'], read['frame_index'])
        if key not in best or read['confidence'] > best[key]['confidence']:
            best[key] = read
    result = []
    for read in previous:
        key = (read.get('segment_id'), read.get('track_id'), read.get('frame_index'))
        if read.get('decoder') == 'digits_only_v4' and key in best:
            replacement = best.pop(key)
            result.append(replacement if replacement['confidence'] > read['confidence'] else read)
        else:
            result.append(read)
    return result + list(best.values())


def recover_masked_reads(run_path, output_path, *, samples_per_track=12, progress=print):
    import cv2
    import torch

    from training.jersey.config import settings
    from training.jersey.crops import candidate
    from training.jersey.reader import ParseqReader
    from training.jersey.roster import Roster

    if samples_per_track < 3:
        raise ValueError('At least three source frames are required')
    if not torch.cuda.is_available():
        raise RuntimeError('Identity recovery requires CUDA; CPU fallback is disabled')
    run_path, output_path = Path(run_path), Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run_path / 'run.json').read_text())
    roster = Roster.load(run_path / 'roster.json')
    votes = json.loads((run_path / 'team_votes.json').read_text())
    if (votes.get('evidence_version') != EVIDENCE_VERSION
            or votes['roster_sha256'] != roster.sha256):
        raise ValueError('Team evidence version or roster does not match')
    named = set()
    with (run_path / 'observations.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            named.update(f"{row['segment_id']}:{person['track_id']}"
                         for person in row['persons'] if person.get('resolved_identity'))
    wanted = defaultdict(set)
    with (run_path / 'jersey_reads.jsonl').open() as source:
        previous = [json.loads(line) for line in source]
    anchors = defaultdict(set)
    for read in previous:
        if (read.get('decoder') == 'digits_only_v4' and read.get('number') is not None
                and read.get('confidence', 0) >= MIN_OCR_CONFIDENCE):
            key = f"{read['segment_id']}:{read['track_id']}"
            anchors[key].update(read['frame_index'] + offset for offset in ANCHOR_OFFSETS)
    for key, track in votes['tracks'].items():
        if key not in named and len(track['frames']) >= 15:
            selected = sample_frames(track['frames'], samples_per_track)
            selected.update(anchors[key].intersection(track['frames']))
            for frame in selected:
                wanted[frame].add(key)
    config = settings({**manifest['config']['jersey'], 'enabled': True,
                       'device': 'cuda:0', 'fallback_device': None,
                       'batch_size': 4, 'mask_policy': 'background_neutral'})
    reader = ParseqReader(config)
    recovered = {}
    pending = []
    crop_count = 0
    started = time.monotonic()

    def flush():
        nonlocal crop_count
        if not pending:
            return
        results = reader.read([item[1]['image'] for item in pending])
        crop_count += len(results)
        for (metadata, crop), result in zip(pending, results, strict=True):
            number = result['number'] if roster.allows(result['number']) else None
            accepted = number is not None and result['confidence'] >= MIN_OCR_CONFIDENCE
            read = {**metadata, 'schema_version': 1, 'run_id': manifest['run_id'],
                    'decoder': 'digits_only_v4', 'sampling': 'mask_overlap_recovery_v1',
                    'processed_frame': metadata['frame_index'],
                    'crop_bbox': crop['bbox'], 'region': crop['region'],
                    'quality': crop['quality'], 'mask_audit': crop['mask_audit'],
                    'bbox_overlap': crop['bbox_overlap'], 'overlap': crop['overlap'],
                    'overlap_method': crop['overlap_method'],
                    'text': result['text'], 'raw_number': result['number'],
                    'number': number, 'confidence': result['confidence'],
                    'status': 'vote' if accepted else 'rejected',
                    'rejection_reason': None if accepted else
                        'not_in_roster' if number is None else 'low_ocr_confidence'}
            key = (read['segment_id'], read['track_id'], read['frame_index'])
            if key not in recovered or recovered[key]['confidence'] < read['confidence']:
                recovered[key] = read
        pending.clear()

    cap = cv2.VideoCapture(manifest['source']['path'])
    if not cap.isOpened():
        raise FileNotFoundError(manifest['source']['path'])
    next_video_frame = 0
    try:
        with (run_path / 'observations.jsonl').open() as source:
            for line in source:
                row = json.loads(line)
                frame_index = row['frame_index']
                if frame_index % 3000 == 0:
                    progress(f'Recovery frame {frame_index}: {crop_count} CUDA crops', flush=True)
                if frame_index not in wanted:
                    continue
                # Sequential grabs avoid imprecise seeking in the original VFR file.
                while next_video_frame <= frame_index:
                    if not cap.grab():
                        raise RuntimeError(f'Video ended before frame {frame_index}')
                    next_video_frame += 1
                ok, image = cap.retrieve()
                if not ok:
                    raise RuntimeError(f'Could not decode frame {frame_index}')
                players = [person for person in row['persons'] if person.get('role') == 'player']
                for person in players:
                    key = f"{row['segment_id']}:{person.get('track_id')}"
                    if key not in wanted[frame_index]:
                        continue
                    episode = next(item for item in votes['tracks'][key]['episodes']
                                   if item['start_frame'] <= frame_index <= item['end_frame'])
                    for variant in ('center', 'upper', 'lower'):
                        crop = candidate(image, person, players, config, variant=variant)
                        if crop is None or crop['bbox_overlap'] <= config['max_overlap']:
                            continue
                        metadata = {'segment_id': row['segment_id'], 'track_id': person['track_id'],
                                    'frame_index': frame_index, 'timestamp_seconds': row['timestamp_s'],
                                    'episode_start_frame': episode['start_frame']}
                        pending.append((metadata, crop))
                        if len(pending) == config['batch_size']:
                            flush()
        flush()
    finally:
        cap.release()
    merged = merge_reads(previous, list(recovered.values()))
    with (output_path / 'jersey_reads.jsonl').open('w') as target:
        for read in merged:
            target.write(json.dumps(read, ensure_ascii=False, separators=(',', ':')) + '\n')
    with (output_path / 'recovered_reads.jsonl').open('w') as target:
        for read in recovered.values():
            target.write(json.dumps(read, ensure_ascii=False) + '\n')
    shutil.copyfile(run_path / 'team_votes.json', output_path / 'team_votes.json')
    refresh_vote_summaries(output_path / 'team_votes.json', output_path / 'jersey_reads.jsonl')
    audit = {'run_id': manifest['run_id'], 'source_sha256': manifest['source']['sha256'],
             'roster_sha256': roster.sha256, 'samples_per_track': samples_per_track,
             'anchor_offsets': ANCHOR_OFFSETS,
             'excluded_named_tracks': len(named), 'crops': crop_count,
             'recovered_source_frames': len(recovered),
             'accepted_source_frames': sum(read['status'] == 'vote' for read in recovered.values()),
             'elapsed_s': time.monotonic() - started, 'ocr': reader.provenance}
    (output_path / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--samples-per-track', type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(recover_masked_reads(args.run, args.output,
                                         samples_per_track=args.samples_per_track), indent=2))
