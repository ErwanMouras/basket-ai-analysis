"""Resolve jersey evidence across a whole run, then replay distance by player.

Raw OCR observations remain intact. Resolved identities are separate annotations so
the viewer can show a player from the first frame of any of their tracks.
"""

from collections import Counter, defaultdict
import bisect
import json
import os
from pathlib import Path

from pipeline.distance import Distance
from pipeline.jersey_votes import (EVIDENCE_VERSION, majority, sample_team as _sample_team,
                                   torso_box as _torso_box,
                                   uniform_signatures as _uniform_signatures)
from training.jersey.colors import color_distance, dominant_color


def _color_timeline(key, observations_path, cap, signatures):
    boxes = {}
    with Path(observations_path).open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            if row['segment_id'] != key[0]:
                continue
            for person in row['persons']:
                if person.get('role') == 'player' and person.get('track_id') == key[1]:
                    boxes[row['frame_index']] = _torso_box(person)
    if not boxes:
        return {}, {}
    import cv2

    first, last = min(boxes), max(boxes)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    raw = {}
    for frame_index in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in boxes:
            raw[frame_index] = _sample_team(frame, boxes[frame_index], signatures)
    return raw, _smooth_colors(raw)


def _smooth_colors(raw):
    known = sorted(frame for frame, team in raw.items() if team)
    smoothed = {}
    for frame_index, team in raw.items():
        if team:
            smoothed[frame_index] = team
            continue
        pos = bisect.bisect_left(known, frame_index)
        before = known[pos - 1] if pos else None
        after = known[pos] if pos < len(known) else None
        if (before is not None and after is not None
                and frame_index - before <= 15 and after - frame_index <= 15
                and raw[before] == raw[after]):
            smoothed[frame_index] = raw[before]
        elif before is not None and after is None and frame_index - before <= 6:
            smoothed[frame_index] = raw[before]
        elif after is not None and before is None and after - frame_index <= 6:
            smoothed[frame_index] = raw[after]
    return smoothed


def load_team_votes(path, roster):
    if path is None or not Path(path).is_file():
        return {}
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if document.get('roster_sha256') != roster.sha256 or document.get('evidence_version') not in (3, 4, 5):
        raise ValueError('Team votes do not match the roster or evidence version')
    result = {}
    for key, item in document['tracks'].items():
        segment, track = map(int, key.split(':'))
        team = item.get('team_id')
        raw = {frame: item['classified'].get(str(frame)) for frame in item['frames']}
        mixed = any(value and value != team for value in raw.values()) if team else False
        allowed = ({frame for frame, value in _smooth_colors(raw).items() if value == team}
                   if mixed else set(item['frames'])) if team else set()
        result[(segment, track)] = {'team_id': team, 'counts': item['counts'],
                                    'share': item['majority_share'], 'raw': raw,
                                    'allowed_frames': allowed, 'mixed': mixed,
                                    'number_majority': item.get('number_majority'),
                                    'episodes': item.get('episodes', [])}
    return result


def track_assignments(reads_path, roster, video_path=None, observations_path=None,
                      team_votes_path=None):
    """Resolve a strict numeric majority against a whole-track color majority."""
    by_track = defaultdict(list)
    modern_tracks = set()
    with Path(reads_path).open(encoding='utf-8') as source:
        for line in source:
            read = json.loads(line)
            if read.get('decoder') in ('digits_only_v3', 'digits_only_v4') and read.get('track_id') is not None:
                modern_tracks.add((read['segment_id'], read['track_id']))
            number = read.get('raw_number') or read.get('number')
            if (read.get('track_id') is not None and number and roster.allows(number)
                    and (read.get('confidence') or 0) >= .8):
                by_track[(read['segment_id'], read['track_id'])].append(read)

    team_votes = load_team_votes(team_votes_path, roster)
    signatures = _uniform_signatures(roster)
    cap = None
    if not team_votes and video_path and signatures:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            cap = None
    assignments = {}
    try:
        for key, reads in by_track.items():
            modern = [read for read in reads if read.get('decoder') == 'digits_only_v4']
            if not modern:
                modern = [read for read in reads if read.get('decoder') == 'digits_only_v3']
            if key in modern_tracks or key in team_votes:
                reads = modern
            if not reads:
                continue
            number, share = majority(
                [read.get('raw_number') or read.get('number') for read in reads],
                min_votes=3 if key in modern_tracks or key in team_votes else 2)
            if number is None:
                continue
            candidates = roster.identity(number)['candidates']
            if not candidates:
                continue
            evidence = team_votes.get(key)
            if evidence:
                team_id = evidence['team_id']
                if team_id is None and (signatures or len(candidates) > 1):
                    continue
                chosen = next((candidate for candidate in candidates
                               if candidate['team_id'] == team_id), None) if team_id else candidates[0]
                if chosen is None or (team_id and chosen['team_id'] != team_id):
                    continue
                # Read frames must belong to the team that won across the track.
                classified = [(frame, color) for frame, color in evidence['raw'].items() if color]
                local = []
                for read in reads:
                    if (read.get('raw_number') or read.get('number')) != number or not classified:
                        continue
                    frame, color = min(classified, key=lambda item: abs(item[0] - read['frame_index']))
                    if abs(frame - read['frame_index']) <= 8:
                        local.append(color)
                local_team, _ = majority(local, min_votes=2)
                if (local_team and local_team != chosen['team_id']) or (evidence['mixed'] and local_team is None):
                    continue
                valid_frames = evidence['allowed_frames'] if evidence['mixed'] else None
                assignments[key] = ({**chosen, '_valid_frames': valid_frames,
                                     '_number_vote_share': share,
                                     '_team_vote_share': evidence['share']}
                                    if valid_frames is not None else
                                    {**chosen, '_number_vote_share': share,
                                     '_team_vote_share': evidence['share']})
                continue
            samples = []
            if cap:
                import cv2

                crop_locations = [(read['frame_index'], read.get('crop_bbox')) for read in reads[:4]]
                for frame_index, bbox in crop_locations:
                    if not bbox:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        signature = dominant_color(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        if signature:
                            samples.append(signature)
            color_votes = Counter()
            for sample in samples:
                matches = [team for team, signature in signatures.items()
                           if signature and color_distance(sample, signature) <= 6]
                if len(matches) == 1:
                    color_votes[matches[0]] += 1
            valid_frames = None
            if len(candidates) == 1:
                chosen = candidates[0]
                if color_votes and chosen['team_id'] not in color_votes:
                    continue
            else:
                if cap and observations_path:
                    raw, smoothed = _color_timeline(key, observations_path, cap, signatures)
                    anchors = Counter()
                    known = [frame for frame, team in raw.items() if team]
                    for read in reads:
                        if not known:
                            break
                        frame = min(known, key=lambda index: abs(index - read['frame_index']))
                        if abs(frame - read['frame_index']) <= 8:
                            anchors[raw[frame]] += 1
                    if len(anchors) == 1 and max(anchors.values()) >= 2:
                        team_id = anchors.most_common(1)[0][0]
                    elif len(color_votes) == 1 and max(color_votes.values()) >= 2:
                        team_id = color_votes.most_common(1)[0][0]
                    else:
                        continue
                    if any(team and team != team_id for team in raw.values()):
                        valid_frames = {frame for frame, team in smoothed.items() if team == team_id}
                        if len(valid_frames) < 2:
                            continue
                elif len(color_votes) == 1 and max(color_votes.values()) >= 2:
                    team_id = color_votes.most_common(1)[0][0]
                else:
                    continue
                chosen = next((candidate for candidate in candidates
                               if candidate['team_id'] == team_id), None)
                if chosen is None:
                    continue
            assignments[key] = {**chosen, '_valid_frames': valid_frames} if valid_frames is not None else chosen
    finally:
        if cap:
            cap.release()
    return assignments


def replay_artifacts(observations_path, tracks_path, reads_path, roster, video_path,
                     distance_config, run_id, team_votes_path=None):
    """Recompute both artifacts atomically before a run is committed."""
    observations_path, tracks_path = Path(observations_path), Path(tracks_path)
    team_votes = load_team_votes(team_votes_path, roster)
    assigned = track_assignments(reads_path, roster, video_path, observations_path,
                                 team_votes_path=team_votes_path)
    from pipeline.spatial_identity import expand_direct_identity_spans, find_spatial_handoffs

    expand_direct_identity_spans(observations_path, assigned, team_votes)
    handoffs = find_spatial_handoffs(observations_path, reads_path, assigned, team_votes)
    handoffs_by_track = defaultdict(list)
    for (segment, track, start), handoff in handoffs.items():
        handoffs_by_track[(segment, track)].append((start, handoff))
    metric = Distance(distance_config, run_id)

    metric.spatial_handoffs = [
        {'segment_id': segment, 'track_id': track, 'start_frame': start,
         **{field: value for field, value in handoff.items() if field != 'player'}}
        for (segment, track, start), handoff in sorted(handoffs.items())]

    def handoff_for(key, frame_index):
        return next((handoff for start, handoff in handoffs_by_track.get(key, ())
                     if start <= frame_index <= handoff['end_frame']), None)

    def player_for(key, frame_index):
        player = assigned.get(key)
        if player is not None:
            valid = player.get('_valid_frames')
            if valid is None or frame_index in valid:
                return player
        handoff = handoff_for(key, frame_index)
        return handoff['player'] if handoff else None

    # Do not merge two simultaneous tracker IDs into one player.
    collisions = set()
    with observations_path.open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            seen = defaultdict(list)
            for person in row['persons']:
                key = (row['segment_id'], person.get('track_id'))
                player = player_for(key, row['frame_index'])
                if person.get('role') == 'player' and player:
                    seen[(player['team_id'], player['number'])].append(key)
            for keys in seen.values():
                if len(keys) > 1:
                    collisions.update((row['frame_index'], key) for key in keys)

    temporary_observations = observations_path.with_name(observations_path.name + '.resolved')
    temporary_tracks = tracks_path.with_name(tracks_path.name + '.resolved')
    from training.jersey.display import display_colors, person_display_color

    palette = display_colors(roster)
    try:
        with observations_path.open(encoding='utf-8') as source, \
             temporary_observations.open('w', encoding='utf-8') as observations, \
             temporary_tracks.open('w', encoding='utf-8') as tracks:
            for line in source:
                row = json.loads(line)
                for person in row['persons']:
                    person.pop('resolved_identity', None)
                    person.pop('jersey_number_suppressed', None)
                    person.pop('resolved_team_id', None)
                    person.pop('majority_jersey_number', None)
                    person.pop('spatial_handoff', None)
                    key = (row['segment_id'], person.get('track_id'))
                    color_vote = team_votes.get(key)
                    if person.get('role') == 'player' and color_vote:
                        person['majority_jersey_number'] = color_vote['number_majority']
                    if (person.get('role') == 'player' and color_vote and color_vote['team_id'] and
                            row['frame_index'] in color_vote['allowed_frames']):
                        person['resolved_team_id'] = color_vote['team_id']
                    player = None if (row['frame_index'], key) in collisions else player_for(key, row['frame_index'])
                    if person.get('role') == 'player' and player:
                        person['resolved_identity'] = {field: value for field, value in player.items()
                                                       if not field.startswith('_')}
                        handoff = handoff_for(key, row['frame_index'])
                        if handoff:
                            person['resolved_identity']['resolution_method'] = 'spatial_handoff'
                            person['spatial_handoff'] = {
                                field: value for field, value in handoff.items()
                                if field not in ('player', 'end_frame')}
                        person['display_color'] = palette['teams'][player['team_id']]
                    else:
                        if (assigned.get(key, {}).get('_valid_frames') is not None
                                or (color_vote and color_vote['mixed']
                                    and row['frame_index'] not in color_vote['allowed_frames'])):
                            person['jersey_number_suppressed'] = True
                        person['display_color'] = (palette['teams'].get(person['resolved_team_id'])
                                                   if person.get('resolved_team_id') else
                                                   palette['unassigned'] if color_vote or person.get('jersey_number_suppressed')
                                                   else person_display_color(person, palette))
                observations.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                for contribution in metric.update(row):
                    tracks.write(json.dumps(contribution, ensure_ascii=False, separators=(',', ':')) + '\n')
        os.replace(temporary_observations, observations_path)
        os.replace(temporary_tracks, tracks_path)
    finally:
        temporary_observations.unlink(missing_ok=True)
        temporary_tracks.unlink(missing_ok=True)
    return metric, len(set(assigned) | set(handoffs_by_track))


def recover_ocr_reads(run_path, roster, targets):
    """Re-read selected tracks from saved detections, without running detection again."""
    import cv2

    from training.jersey.config import number, settings
    from training.jersey.crops import candidate
    from training.jersey.reader import ParseqReader

    run_path = Path(run_path)
    manifest = json.loads((run_path / 'run.json').read_text(encoding='utf-8'))
    reads_path = run_path / 'jersey_reads.jsonl'
    with reads_path.open(encoding='utf-8') as source:
        existing = [json.loads(line) for line in source]
    frames = {key: set(extra) for key, extra in targets.items()}
    for read in existing:
        key = (read.get('segment_id'), read.get('track_id'))
        if key in frames:
            frames[key].add(read['frame_index'])
    already = {(read.get('segment_id'), read.get('track_id'), read.get('frame_index'))
               for read in existing if read.get('region') == 'pose_number'}
    observations = {}
    with (run_path / 'observations.jsonl').open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            for person in row['persons']:
                key = (row['segment_id'], person.get('track_id'))
                if (key in frames and row['frame_index'] in frames[key]
                        and (key[0], key[1], row['frame_index']) not in already):
                    observations[(key[0], key[1], row['frame_index'])] = (row, person)
    cap = cv2.VideoCapture(manifest['source']['path'])
    if not cap.isOpened():
        raise FileNotFoundError(manifest['source']['path'])
    config = settings({'enabled': True})
    crops = []
    try:
        for (segment, track, frame_index), (row, person) in sorted(observations.items(),
                                                                     key=lambda item: item[0][2]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = cap.read()
            if not ok:
                continue
            crop = candidate(image, person,
                             [other for other in row['persons'] if other.get('role') == 'player'], config)
            if crop is not None and crop['region'] == 'pose_number':
                crops.append((segment, track, row, crop))
    finally:
        cap.release()
    if not crops:
        return 0
    reader = ParseqReader(config)
    recovered = []
    for start in range(0, len(crops), config['batch_size']):
        batch = crops[start:start + config['batch_size']]
        results = reader.read([crop['image'] for _, _, _, crop in batch])
        for (segment, track, row, crop), result in zip(batch, results, strict=True):
            raw = number(result['text']) if result['eos'] else None
            permitted = raw is not None and roster.allows(raw)
            accepted = permitted and result['confidence'] >= config['min_ocr_confidence']
            reason = ('invalid_text' if raw is None else 'not_in_roster' if not permitted
                      else 'low_ocr_confidence' if not accepted else None)
            recovered.append({'schema_version': 1, 'run_id': manifest['run_id'],
                'track_id': track, 'segment_id': segment, 'processed_frame': row['frame_index'],
                'frame_index': row['frame_index'], 'timestamp_seconds': row['timestamp_s'],
                'crop_bbox': crop['bbox'], 'region': crop['region'], 'quality': crop['quality'],
                'text': result['text'], 'number': raw if permitted else None,
                'raw_number': raw, 'rejection_reason': reason,
                'confidence': result['confidence'], 'status': 'vote' if accepted else 'rejected'})
    temporary = reads_path.with_name(reads_path.name + '.rescued')
    try:
        with temporary.open('w', encoding='utf-8') as output:
            for read in existing + recovered:
                output.write(json.dumps(read, ensure_ascii=False, separators=(',', ':')) + '\n')
        os.replace(temporary, reads_path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(recovered)


def enrich_completed_run(run_path, roster_path, *, rescue_tracks=None, rebuild_votes=False):
    """Apply the same resolution to an older completed run without model inference."""
    from pipeline.contracts import validate_statistics
    from training.common.files import write_json
    from training.common.provenance import file_hash
    from training.jersey.display import display_colors
    from training.jersey.roster import Roster

    run_path = Path(run_path)
    roster = Roster.load(roster_path)
    manifest_path = run_path / 'run.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') not in ('completed', 'partial'):
        raise ValueError('Run is not completed')
    observations = run_path / 'observations.jsonl'
    if file_hash(observations) != manifest['artifacts']['observations.jsonl']['sha256']:
        raise ValueError('Observation artifact hash mismatch')
    original_roster_sha = (manifest.get('models', {}).get('jersey', {}).get('roster', {})
                           .get('sha256', manifest.get('roster_sha256')))
    prior_resolution_sha = manifest.get('identity_resolution', {}).get('roster_sha256')
    if ((original_roster_sha and original_roster_sha != roster.sha256)
            or (prior_resolution_sha and prior_resolution_sha != roster.sha256)):
        raise ValueError('Run has a different roster snapshot')
    rescued = recover_ocr_reads(run_path, roster, rescue_tracks) if rescue_tracks else 0
    team_votes_path = run_path / 'team_votes.json'
    existing_version = (json.loads(team_votes_path.read_text(encoding='utf-8'))
                        .get('evidence_version') if team_votes_path.is_file() else None)
    built_votes = rebuild_votes or ((existing_version != EVIDENCE_VERSION)
                                    and manifest.get('capabilities', {}).get('jersey') == 'ok')
    if built_votes:
        from pipeline.jersey_votes import build_track_votes

        manifest['track_vote_evidence'] = build_track_votes(
            observations, run_path / 'jersey_reads.jsonl', manifest['source']['path'],
            roster, manifest['run_id'], team_votes_path, manifest['config']['jersey'])
    elif team_votes_path.is_file():
        from pipeline.jersey_votes import refresh_vote_summaries

        refresh_vote_summaries(team_votes_path, run_path / 'jersey_reads.jsonl')
    metric, resolved = replay_artifacts(
        observations, run_path / 'tracks.jsonl', run_path / 'jersey_reads.jsonl',
        roster, manifest['source']['path'], manifest['config']['distance'], manifest['run_id'],
        team_votes_path=team_votes_path if team_votes_path.is_file() else None)
    previous = json.loads((run_path / 'statistics.json').read_text(encoding='utf-8'))
    stats = {**previous, **metric.result()}
    validate_statistics(stats)
    write_json(run_path / 'statistics.json', stats)
    write_json(run_path / 'roster.json', roster.payload)
    manifest['roster_sha256'] = original_roster_sha
    total_rescued = manifest.get('identity_resolution', {}).get('rescued_ocr_reads', 0) + rescued
    manifest['identity_resolution'] = {
        'mode': 'posthoc', 'roster_sha256': roster.sha256,
        'number_evidence': 'digit-only OCR every two frames until confirmed by episode majority',
        'shared_number_evidence': 'strict majority of classified torso colors across every track frame',
    }
    if total_rescued:
        manifest['identity_resolution']['rescued_ocr_reads'] = total_rescued
    manifest['display_colors'] = display_colors(roster)
    manifest['resolved_player_tracks'] = resolved
    manifest['spatial_handoffs'] = metric.spatial_handoffs
    artifacts = ['observations.jsonl', 'tracks.jsonl', 'jersey_reads.jsonl', 'statistics.json']
    if team_votes_path.is_file():
        artifacts.append('team_votes.json')
    for name in artifacts:
        manifest['artifacts'][name] = {'sha256': file_hash(run_path / name)}
    write_json(manifest_path, manifest)
    return {'resolved_tracks': resolved, 'players': len(stats['players']), 'rescued_ocr_reads': rescued}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Resolve jersey identities in a completed run')
    parser.add_argument('run')
    parser.add_argument('roster')
    parser.add_argument('--rescue-track', action='append', default=[], metavar='SEGMENT:TRACK[:FRAME]')
    parser.add_argument('--rebuild-votes', action='store_true')
    args = parser.parse_args()
    targets = defaultdict(set)
    for value in args.rescue_track:
        pieces = value.split(':')
        if len(pieces) not in (2, 3) or any(not piece.isdecimal() for piece in pieces):
            parser.error('--rescue-track requires SEGMENT:TRACK[:FRAME]')
        targets[(int(pieces[0]), int(pieces[1]))].update([int(pieces[2])] if len(pieces) == 3 else [])
    print(json.dumps(enrich_completed_run(args.run, args.roster, rescue_tracks=targets,
                                          rebuild_votes=args.rebuild_votes), ensure_ascii=False))
