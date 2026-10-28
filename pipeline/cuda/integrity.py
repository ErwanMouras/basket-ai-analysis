"""ID/sequence constraints are host metadata; timestamp arithmetic stays CUDA."""

import torch

from pipeline.cuda.core import tensor


def observation_metadata(row, run_id, index):
    if row["frame_index"] != index or row["run_id"] != run_id:
        raise ValueError(f"Observation sequence/run mismatch at frame {index}")
    tracks, identities, observations = set(), set(), set()
    for detection, person in enumerate(row["persons"]):
        if "observation_id" in person:
            expected = f"{run_id}:{index}:{detection}"
            if person["observation_id"] != expected or expected in observations:
                raise ValueError(f"Observation ID mismatch at frame {index}")
            observations.add(expected)
        track = person["track_id"]
        if track is not None:
            if type(track) is not int or track < 1 or track in tracks:
                raise ValueError(f"Duplicate/invalid track ID at frame {index}")
            tracks.add(track)
        identity = person.get("resolved_identity")
        if identity:
            key = (identity["team_id"], identity["number"])
            if person.get("jersey_number_suppressed") or key in identities:
                raise ValueError(f"Conflicting resolved identity at frame {index}")
            identities.add(key)


def timestamps(values, device, *, previous=None, expected=None):
    actual = tensor(values, device=device)
    valid = (
        torch.isfinite(actual).all() & (actual >= 0).all() & (actual.diff() > 0).all()
    )
    if previous is not None:
        valid &= actual[0] > previous
    if expected is not None:
        if actual.shape != expected.shape:
            raise ValueError("Source timestamp count mismatch")
        valid &= (actual == expected).all()
    if not bool(valid):
        raise ValueError(
            "Observation timestamps are nonfinite, unordered or differ from source PTS"
        )
    return actual[-1]
