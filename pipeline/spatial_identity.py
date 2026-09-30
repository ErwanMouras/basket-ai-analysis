"""Conservative identity handoffs between nearby, non-overlapping track spans."""

from collections import defaultdict
from collections import Counter
import json
import math
from pathlib import Path

from pipeline.jersey_votes import split_color_episodes


MAX_GAP_S = 0.20
MAX_PIXEL_DISTANCE_IN_HEIGHTS = 0.55
MIN_NEIGHBOR_DISTANCE_IN_HEIGHTS = 0.65
MIN_SPAN_FRAMES = 4


def _pixel_distance(first, second):
    a, b = first.get('bbox'), second.get('bbox')
    if not a or not b:
        return None
    heights = [box[3] - box[1] for box in (a, b)]
    if min(heights) <= 0:
        return None
    foot_a = ((a[0] + a[2]) / 2, a[3])
    foot_b = ((b[0] + b[2]) / 2, b[3])
    return math.dist(foot_a, foot_b) / max(heights)


def _ground_distance(first, second):
    a, b = first.get('position_m'), second.get('position_m')
    return math.dist(a, b) if a is not None and b is not None else None


def _span_team(span, team_votes, *, allow_pure_track_fallback=False):
    evidence = team_votes.get(span['key'])
    if not evidence:
        return None, Counter()
    counts = Counter(evidence['raw'].get(frame) for frame, _, _ in span['points'])
    counts.pop(None, None)
    total = sum(counts.values())
    if total >= 3:
        team, count = counts.most_common(1)[0]
        return (team if count / total >= .8 else None), counts
    if (allow_pure_track_fallback and total == 0 and evidence['team_id']
            and sum(evidence['counts'].values()) >= 5
            and evidence['counts'].get(evidence['team_id'], 0) == sum(evidence['counts'].values())):
        return evidence['team_id'], counts
    return None, counts


def _neighbor_team(person, row, span_at, team_votes):
    track = person.get('track_id')
    if track is None:
        return None
    span = span_at.get((row['frame_index'], row['segment_id'], track))
    if not span:
        return None
    team, local = _span_team(span, team_votes, allow_pure_track_fallback=True)
    if team:
        return team
    evidence = team_votes.get(span['key'], {})
    global_counts = evidence.get('counts', {})
    known = sum(global_counts.values())
    if (sum(local.values()) >= 2 and len(local) == 1
            and evidence.get('team_id') in local
            and known >= 5 and global_counts[evidence['team_id']] / known >= .9):
        return evidence['team_id']
    return None


def _box_iou(first, second):
    a, b = first['bbox'], second['bbox']
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap) if area_a + area_b > overlap else 0


def _crowded(person, row, span_at, team_votes, expected_team,
             *, allow_untracked_duplicate=False):
    for other in row['persons']:
        if other is person or other.get('role') != 'player':
            continue
        pixels = _pixel_distance(person, other)
        neighbor = _neighbor_team(other, row, span_at, team_votes)
        if (pixels is not None and pixels < MIN_NEIGHBOR_DISTANCE_IN_HEIGHTS
                and (neighbor is None or neighbor == expected_team)):
            if (allow_untracked_duplicate and other.get('track_id') is None
                    and pixels < .12 and _box_iou(person, other) >= .5):
                continue
            return True
    return False


def _crowded_gap(first, last, rows, end_frame, start_frame, span_at, team_votes, expected_team):
    """Reject a body crossing the short corridor while the tracker is absent."""
    a, b = first['bbox'], last['bbox']
    foot_a, foot_b = ((a[0] + a[2]) / 2, a[3]), ((b[0] + b[2]) / 2, b[3])
    for frame in range(end_frame + 1, start_frame):
        fraction = (frame - end_frame) / (start_frame - end_frame)
        expected = tuple(x + fraction * (y - x) for x, y in zip(foot_a, foot_b))
        for person in rows[frame]['persons']:
            if person.get('role') != 'player' or not person.get('bbox'):
                continue
            box = person['bbox']
            height = box[3] - box[1]
            neighbor = _neighbor_team(person, rows[frame], span_at, team_votes)
            if (height > 0
                    and math.dist(expected, ((box[0] + box[2]) / 2, box[3])) / height < MIN_NEIGHBOR_DISTANCE_IN_HEIGHTS
                    and (neighbor is None or neighbor == expected_team)):
                return True
    return False


def _track_spans(rows, team_votes=None):
    observations = defaultdict(list)
    for row in rows.values():
        for person in row['persons']:
            if person.get('role') == 'player' and person.get('track_id') is not None:
                observations[(row['segment_id'], person['track_id'])].append(
                    (row['frame_index'], row['timestamp_s'], person))
    spans = []
    for key, points in observations.items():
        raw = (team_votes or {}).get(key, {}).get('raw', {})
        by_frame = {frame: (frame, timestamp, person) for frame, timestamp, person in points}
        for frames in split_color_episodes([frame for frame, _, _ in points], raw):
            episode = [by_frame[frame] for frame in frames]
            spans.append({'key': key, 'points': episode,
                          'start': frames[0], 'end': frames[-1]})
    return spans


def expand_direct_identity_spans(observations_path, assignments, team_votes):
    """Keep a direct identity through unclassified frames of one clean span."""
    if not team_votes:
        return
    rows = {}
    with Path(observations_path).open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            rows[row['frame_index']] = row
    for span in _track_spans(rows, team_votes):
        key = span['key']
        player = assignments.get(key)
        if not player or '_valid_frames' not in player:
            continue
        valid = player['_valid_frames']
        if not any(frame in valid for frame, _, _ in span['points']):
            continue
        _, colors = _span_team(span, team_votes)
        if colors.get(player['team_id'], 0) < 5 or any(
                team != player['team_id'] for team in colors):
            continue
        for frame, _, _ in span['points']:
            duplicate = any(other_key != key
                and other_player['team_id'] == player['team_id']
                and other_player['number'] == player['number']
                and (other_player.get('_valid_frames') is None
                     or frame in other_player['_valid_frames'])
                for other in rows[frame]['persons']
                if other.get('role') == 'player' and other.get('track_id') is not None
                for other_key in [(rows[frame]['segment_id'], other['track_id'])]
                for other_player in [assignments.get(other_key)] if other_player)
            if not duplicate:
                valid.add(frame)


def find_spatial_handoffs(observations_path, reads_path, assignments, team_votes):
    """Return {(segment, track, first_frame): safe handoff} for complete spans.

    The source needs a direct identity supported by its local color evidence.
    Nearby teammates or unknown bodies block a match; confirmed opponents can
    cross the image without hiding a same-team handoff. Only a conflict-free
    prefix of the target span can inherit the identity.
    """
    rows = {}
    with Path(observations_path).open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            rows[row['frame_index']] = row
    spans = _track_spans(rows, team_votes)
    starts = defaultdict(list)
    span_at = {}
    for span in spans:
        starts[span['key'][0]].append(span)
        for frame, _, _ in span['points']:
            span_at[(frame, *span['key'])] = span
    for group in starts.values():
        group.sort(key=lambda span: span['start'])

    reliable_reads = defaultdict(list)
    all_reads = []
    with Path(reads_path).open(encoding='utf-8') as source:
        for line in source:
            all_reads.append(json.loads(line))
    decoder = ('digits_only_v4' if any(read.get('decoder') == 'digits_only_v4'
                                       for read in all_reads) else 'digits_only_v3')
    for read in all_reads:
        key = (read.get('segment_id'), read.get('track_id'))
        if ((read.get('decoder') == decoder if team_votes else read.get('track_id') is not None)
                and read.get('number') is not None and read.get('confidence', 0) >= .8):
            reliable_reads[key].append(read)

    candidates = []
    for source in spans:
        key = source['key']
        player = assignments.get(key)
        if player is None or len(source['points']) < 2:
            continue
        valid = player.get('_valid_frames')
        source_team, source_colors = _span_team(source, team_votes)
        if len(source['points']) < MIN_SPAN_FRAMES and source_colors.get(player['team_id'], 0) < 1:
            continue
        if source_team not in (None, player['team_id']):
            continue
        if any(team != player['team_id'] for team in source_colors):
            continue
        if (valid is not None and source_team != player['team_id']
                and any(frame not in valid for frame, _, _ in source['points'][-2:])):
            continue
        end_frame, end_time, end_person = source['points'][-1]
        end_row = rows[end_frame]
        if not end_row.get('included', True):
            continue
        identity = (player['team_id'], player['number'])
        if any(other_key != key and other_player['team_id'] == identity[0]
               and other_player['number'] == identity[1]
               and (other_player.get('_valid_frames') is None
                    or end_frame in other_player['_valid_frames'])
               for other in end_row['persons']
               if other.get('role') == 'player' and other.get('track_id') is not None
               for other_key in [(end_row['segment_id'], other['track_id'])]
               for other_player in [assignments.get(other_key)] if other_player):
            continue
        for target in starts[key[0]]:
            if target['key'] == key:
                continue
            start_frame, start_time, start_person = target['points'][0]
            direct = assignments.get(target['key'])
            if direct and (direct.get('_valid_frames') is None
                           or any(frame in direct['_valid_frames']
                                  for frame, _, _ in target['points'])):
                continue
            gap = start_time - end_time
            if not 0 < gap <= MAX_GAP_S or len(target['points']) < MIN_SPAN_FRAMES:
                continue
            if any(rows[frame].get('scene_cut') for frame in range(end_frame + 1, start_frame + 1)):
                continue
            pixels = _pixel_distance(end_person, start_person)
            if pixels is None or pixels > min(MAX_PIXEL_DISTANCE_IN_HEIGHTS, .35 + gap):
                continue
            heights = [(person['bbox'][3] - person['bbox'][1]) for person in (end_person, start_person)]
            if min(heights) / max(heights) < .65:
                continue
            meters = _ground_distance(end_person, start_person)
            evidence = team_votes.get(target['key'])
            local_counts = Counter()
            if evidence:
                local_team, local_counts = _span_team(target, team_votes)
                if local_team not in (None, player['team_id']):
                    continue
                if any(color != player['team_id'] for color in local_counts):
                    continue
                if (local_team is None and local_counts.get(player['team_id'], 0) < 3
                        and evidence['team_id'] not in (None, player['team_id'])):
                    continue
            if _crowded(end_person, end_row, span_at, team_votes, player['team_id'],
                        allow_untracked_duplicate=(pixels < .12 and
                            local_counts.get(player['team_id'], 0) >= 3)):
                continue
            start_row = rows[start_frame]
            if (not start_row.get('included', True)
                    or _crowded(start_person, start_row, span_at, team_votes, player['team_id'])):
                continue
            if _crowded_gap(end_person, start_person, rows, end_frame, start_frame,
                            span_at, team_votes, player['team_id']):
                continue
            if any(start_frame <= read['frame_index'] <= target['end']
                   and (read.get('number') or read.get('raw_number')) != player['number']
                   for read in reliable_reads[target['key']]):
                continue
            # Stop just before a directly identified copy reappears. Tracker IDs
            # may briefly duplicate the same body during a handoff.
            safe_points = []
            for frame, timestamp, person in target['points']:
                duplicate = any(other_key != target['key']
                    and other_player['team_id'] == identity[0]
                    and other_player['number'] == identity[1]
                    and (other_player.get('_valid_frames') is None
                         or frame in other_player['_valid_frames'])
                    for other in rows[frame]['persons']
                    if other.get('role') == 'player' and other.get('track_id') is not None
                    for other_key in [(rows[frame]['segment_id'], other['track_id'])]
                    for other_player in [assignments.get(other_key)] if other_player)
                if duplicate:
                    break
                safe_points.append((frame, timestamp, person))
            if len(safe_points) < MIN_SPAN_FRAMES:
                continue
            target = {**target, 'points': safe_points, 'end': safe_points[-1][0]}
            candidates.append({'source': source, 'target': target, 'player': player,
                               'gap_s': round(gap, 4), 'distance_m': meters,
                               'distance_in_heights': round(pixels, 4)})

    by_source, by_target = defaultdict(list), defaultdict(list)
    for item in candidates:
        by_source[(item['source']['key'], item['source']['end'])].append(item)
        by_target[(item['target']['key'], item['target']['start'])].append(item)
    accepted = {}
    for item in candidates:
        source, target = item['source'], item['target']
        if (len(by_source[(source['key'], source['end'])]) != 1
                or len(by_target[(target['key'], target['start'])]) != 1):
            continue
        accepted[(target['key'][0], target['key'][1], target['start'])] = {
            'end_frame': target['end'], 'player': item['player'],
            'direction': 'forward',
            'source_segment_id': source['key'][0], 'source_track_id': source['key'][1],
            'source_end_frame': source['end'], 'gap_s': item['gap_s'],
            'distance_m': round(item['distance_m'], 4) if item['distance_m'] is not None else None,
            'distance_in_heights': item['distance_in_heights'],
        }
    reverse = _reverse_handoffs(rows, spans, span_at, reliable_reads,
                                assignments, team_votes, accepted)
    for key, item in reverse.items():
        identity = (item['player']['team_id'], item['player']['number'])
        if any((link['player']['team_id'], link['player']['number']) == identity
               and key[0] == link_key[0]
               and key[2] <= link['end_frame'] and link_key[2] <= item['end_frame']
               for link_key, link in accepted.items()):
            continue
        accepted[key] = item
    # Two inferred spans for the same player cannot coexist on one frame.
    conflicting = set()
    for first_key, first in accepted.items():
        for second_key, second in accepted.items():
            if first_key >= second_key or first_key[0] != second_key[0]:
                continue
            if ((first['player']['team_id'], first['player']['number']) ==
                    (second['player']['team_id'], second['player']['number'])
                    and first_key[2] <= second['end_frame']
                    and second_key[2] <= first['end_frame']):
                conflicting.update((first_key, second_key))
    for key in conflicting:
        del accepted[key]
    return accepted


def _reverse_handoffs(rows, spans, span_at, reliable_reads, assignments,
                      team_votes, forward):
    """Inherit a later direct identity on the safe suffix of an earlier span."""
    candidates = []
    for source in spans:
        player = assignments.get(source['key'])
        if not player or len(source['points']) < 2:
            continue
        start_frame, start_time, start_person = source['points'][0]
        valid = player.get('_valid_frames')
        if valid is not None and start_frame not in valid:
            continue
        source_team, source_colors = _span_team(source, team_votes)
        if (source_team not in (None, player['team_id'])
                or any(team != player['team_id'] for team in source_colors)):
            continue
        start_row = rows[start_frame]
        if not start_row.get('included', True):
            continue
        identity = (player['team_id'], player['number'])
        if any(other_key != source['key']
               and other_player['team_id'] == identity[0]
               and other_player['number'] == identity[1]
               and (other_player.get('_valid_frames') is None
                    or start_frame in other_player['_valid_frames'])
               for other in start_row['persons']
               if other.get('role') == 'player' and other.get('track_id') is not None
               for other_key in [(start_row['segment_id'], other['track_id'])]
               for other_player in [assignments.get(other_key)] if other_player):
            continue

        for target in spans:
            if (target['key'] == source['key']
                    or target['key'][0] != source['key'][0]
                    or len(target['points']) < MIN_SPAN_FRAMES):
                continue
            end_frame, end_time, end_person = target['points'][-1]
            gap = start_time - end_time
            if not 0 < gap <= MAX_GAP_S:
                continue
            # A tracker ID still visible at the handoff frame represents a
            # second body or an unresolved ID swap, not a clean disappearance.
            if any(person.get('role') == 'player'
                   and person.get('track_id') == target['key'][1]
                   for frame in range(end_frame + 1, start_frame + 1)
                   for person in rows[frame]['persons']
                   if rows[frame]['segment_id'] == target['key'][0]):
                continue
            direct = assignments.get(target['key'])
            if direct and (direct.get('_valid_frames') is None
                           or any(frame in direct['_valid_frames']
                                  for frame, _, _ in target['points'])):
                continue
            if any(link_key[:2] == target['key']
                   and link_key[2] <= target['end'] and target['start'] <= link['end_frame']
                   for link_key, link in forward.items()):
                continue
            if any(rows[frame].get('scene_cut')
                   for frame in range(end_frame + 1, start_frame + 1)):
                continue
            pixels = _pixel_distance(end_person, start_person)
            if pixels is None or pixels > min(MAX_PIXEL_DISTANCE_IN_HEIGHTS, .35 + gap):
                continue
            heights = [person['bbox'][3] - person['bbox'][1]
                       for person in (end_person, start_person)]
            if min(heights) <= 0 or min(heights) / max(heights) < .65:
                continue
            evidence = team_votes.get(target['key'])
            local_counts = Counter()
            if evidence:
                local_team, local_counts = _span_team(target, team_votes)
                if local_team not in (None, player['team_id']):
                    continue
                if any(color != player['team_id'] for color in local_counts):
                    continue
                if (local_team is None and local_counts.get(player['team_id'], 0) < 3
                        and evidence['team_id'] not in (None, player['team_id'])):
                    continue
            end_row = rows[end_frame]
            if (not end_row.get('included', True)
                    or _crowded(end_person, end_row, span_at, team_votes, player['team_id'],
                                allow_untracked_duplicate=(pixels < .12 and
                                    local_counts.get(player['team_id'], 0) >= 3))
                    or _crowded(start_person, start_row, span_at, team_votes,
                                player['team_id'])):
                continue
            if _crowded_gap(end_person, start_person, rows, end_frame, start_frame,
                            span_at, team_votes, player['team_id']):
                continue
            if any(target['start'] <= read['frame_index'] <= target['end']
                   and (read.get('number') or read.get('raw_number')) != player['number']
                   for read in reliable_reads[target['key']]):
                continue

            safe_points = []
            for frame, timestamp, person in reversed(target['points']):
                duplicate = any(other_key != target['key']
                    and other_player['team_id'] == identity[0]
                    and other_player['number'] == identity[1]
                    and (other_player.get('_valid_frames') is None
                         or frame in other_player['_valid_frames'])
                    for other in rows[frame]['persons']
                    if other.get('role') == 'player' and other.get('track_id') is not None
                    for other_key in [(rows[frame]['segment_id'], other['track_id'])]
                    for other_player in [assignments.get(other_key)] if other_player)
                if duplicate:
                    break
                safe_points.append((frame, timestamp, person))
            if len(safe_points) < MIN_SPAN_FRAMES:
                continue
            safe_points.reverse()
            candidates.append({'source': source, 'target': target,
                               'safe_points': safe_points, 'player': player,
                               'gap_s': round(gap, 4),
                               'distance_m': _ground_distance(end_person, start_person),
                               'distance_in_heights': round(pixels, 4)})

    by_source, by_target = defaultdict(list), defaultdict(list)
    for item in candidates:
        by_source[(item['source']['key'], item['source']['start'])].append(item)
        by_target[(item['target']['key'], item['target']['end'])].append(item)
    accepted = {}
    for item in candidates:
        source, target = item['source'], item['target']
        if (len(by_source[(source['key'], source['start'])]) != 1
                or len(by_target[(target['key'], target['end'])]) != 1):
            continue
        first = item['safe_points'][0][0]
        accepted[(target['key'][0], target['key'][1], first)] = {
            'end_frame': target['end'], 'player': item['player'],
            'direction': 'backward',
            'source_segment_id': source['key'][0], 'source_track_id': source['key'][1],
            'source_start_frame': source['start'], 'gap_s': item['gap_s'],
            'distance_m': (round(item['distance_m'], 4)
                           if item['distance_m'] is not None else None),
            'distance_in_heights': item['distance_in_heights'],
        }
    return accepted
