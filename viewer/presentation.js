// Display-only stability. Source observations, identities and metrics stay intact.
(function (root) {
  'use strict';
  function identified(person) {
    if (person.jersey_number_suppressed) return false;
    if (person.resolved_identity?.team_id && person.resolved_identity.number != null) return true;
    if (Object.hasOwn(person, 'majority_jersey_number')) return false;
    const jersey = person.jersey || {};
    return jersey.status === 'confirmed' && jersey.identity_status === 'unique_number'
      && Boolean(jersey.team_id) && jersey.number != null;
  }
  const key = (row, person) => `${row.segment_id ?? 0}:${person.role}:${person.track_id}`;
  function spanLength(row, person, spans) {
    if (!Number.isInteger(row.frame_index)) return null;
    if (person.track_id == null) return 0;
    const intervals = spans?.[key(row, person)];
    if (!intervals) return null; // Compatible with older servers and imported runs.
    let lo = 0, hi = intervals.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (intervals[mid][0] <= row.frame_index) lo = mid + 1; else hi = mid;
    }
    const interval = intervals[lo - 1];
    return interval && row.frame_index <= interval[1] ? interval[1] - interval[0] + 1 : 0;
  }
  function visible(row, person, spans, minimum = 5) {
    if (!minimum || identified(person)) return true;
    const count = spanLength(row, person, spans);
    return count == null || count >= minimum;
  }
  const validBox = box => Array.isArray(box) && box.length === 4 && box.every(Number.isFinite)
    && box[2] > box[0] && box[3] > box[1];
  function compatible(first, second) {
    if (first.role !== second.role || first.track_id !== second.track_id || !validBox(second.bbox)) return false;
    if (identified(first) && identified(second)) {
      const a = first.resolved_identity || first.jersey, b = second.resolved_identity || second.jersey;
      if (a.team_id !== b.team_id || a.number !== b.number) return false;
    }
    const a = first.bbox, b = second.bbox;
    const height = Math.max(a[3] - a[1], b[3] - b[1]);
    const displacement = Math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2);
    const ratio = (b[3] - b[1]) / (a[3] - a[1]);
    return displacement <= .65 * height && ratio >= .65 && ratio <= 1.55;
  }
  function smoothBox(row, person, frameAt) {
    if (!validBox(person.bbox) || !Number.isInteger(row.frame_index) || person.track_id == null || row.scene_cut) return person.bbox;
    const sum = person.bbox.map(value => value * 3);
    let weight = 3;
    // Symmetric pairs keep constant-velocity movement aligned with this image.
    // No estimates are carried across missing observations or camera cuts.
    for (let radius = 1; radius <= 2; radius++) {
      const neighborhood = [];
      let safe = true;
      for (let offset = -radius; offset <= radius; offset++) {
        const frame = offset ? frameAt(row.frame_index + offset) : row;
        if (!frame || frame.segment_id !== row.segment_id || frame.width !== row.width || frame.height !== row.height
            || (offset > -radius && frame.scene_cut)) { safe = false; break; }
        const other = offset ? frame.persons?.find(p => p.role === person.role && p.track_id === person.track_id) : person;
        if (!other || !compatible(person, other)) { safe = false; break; }
        neighborhood.push(other);
      }
      if (!safe) break;
      const factor = radius === 1 ? 2 : 1;
      for (let coordinate = 0; coordinate < 4; coordinate++)
        sum[coordinate] += factor * (neighborhood[0].bbox[coordinate] + neighborhood.at(-1).bbox[coordinate]);
      weight += 2 * factor;
    }
    return weight === 3 ? person.bbox : sum.map(value => value / weight);
  }
  function prepare(row, spans, frameAt, options = {}) {
    return {...row, persons: (row.persons || []).filter(p =>
      visible(row, p, spans, options.stableTracks ? 5 : 0)
    ).map(p => options.smoothBoxes ? {...p, bbox: smoothBox(row, p, frameAt)} : p)};
  }
  const api = {identified, key, spanLength, visible, smoothBox, prepare};
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.VideoPresentation = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
