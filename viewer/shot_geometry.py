"""Small projective helpers for shot evidence; no numeric runtime dependency.

A floor homography stabilizes the image, not the airborne ball in metres. The
rim anchor estimates a square-pixel camera with principal point at image centre.
It is a broad image-space gate, never a measured 3D ball position.
"""

import math


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def point(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(map(number, value))


def matrix(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(
            isinstance(row, (list, tuple)) and len(row) == 3 and all(map(number, row))
            for row in value
        )
    )


def inverse(m):
    if not matrix(m):
        return None
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    adj = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ]
    det = a * adj[0][0] + b * adj[1][0] + c * adj[2][0]
    scale = max(abs(v) for row in m for v in row)
    if not scale or abs(det) < 1e-15 * scale**3:
        return None
    return [[v / det for v in row] for row in adj]


def multiply(m, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in m)


def project(m, xy):
    if not matrix(m) or not point(xy):
        return None
    x, y, z = multiply(m, (*xy, 1))
    return (x / z, y / z) if all(map(number, (x, y, z))) and abs(z) > 1e-9 else None


def warp(reference_inverse, current, xy):
    floor = project(current, xy)
    return project(reference_inverse, floor) if floor else None


def rim_anchors(court, geometry, width, height):
    """Approximate elevated NBA rims from a plausible calibrated camera."""
    if not geometry or geometry.get("league") != "NBA" or not number(width) or not number(height):
        return []
    h = inverse(court)
    if h is None:
        return []
    cx, cy = width / 2, height / 2
    a = [h[0][0] - cx * h[2][0], h[1][0] - cy * h[2][0]]
    b = [h[0][1] - cx * h[2][1], h[1][1] - cy * h[2][1]]
    c, d = h[2][:2]
    aa = [sum(x * y for x, y in zip(a, b)), sum(x * x - y * y for x, y in zip(a, b))]
    bb = [c * d, c * c - d * d]
    denom = sum(x * x for x in aa)
    inv_f2 = -sum(x * y for x, y in zip(aa, bb)) / denom if denom else -1
    if inv_f2 <= 0:
        return []
    focal = 1 / math.sqrt(inv_f2)
    if not 0.4 * width <= focal <= 8 * width:
        return []
    q = [
        [(h[0][j] - cx * h[2][j]) / focal, (h[1][j] - cy * h[2][j]) / focal, h[2][j]]
        for j in (0, 1)
    ]
    norms = [math.sqrt(sum(v * v for v in col)) for col in q]
    if min(norms) <= 0 or max(norms) / min(norms) > 1.3:
        return []
    if abs(sum(x * y for x, y in zip(*q))) / (norms[0] * norms[1]) > 0.2:
        return []
    scale = 2 / sum(norms)
    u, v = [[x * scale for x in col] for col in q]
    cross = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    vertical = [
        (focal * cross[0] + cx * cross[2]) / scale,
        (focal * cross[1] + cy * cross[2]) / scale,
        cross[2] / scale,
    ]
    length, breadth = geometry.get("length"), geometry.get("width")
    if not all(number(v) and v > 0 for v in (length, breadth)):
        return []
    rims = []
    for x in (1.6002, length - 1.6002):
        base = multiply(h, (x, breadth / 2, 1))
        choices = []
        for sign in (-1, 1):
            p = [a + sign * 3.048 * b for a, b in zip(base, vertical)]
            if p[2] * base[2] > 0 and abs(p[2]) > 1e-9:
                choices.append((p[0] / p[2], p[1] / p[2]))
        if choices:
            rim = min(choices, key=lambda p: p[1])
            if -0.1 * width <= rim[0] <= 1.1 * width and 0 <= rim[1] <= height:
                rims.append(rim)
    return rims


def quadratic(times, values):
    """Least-squares quadratic in elapsed seconds, returning coefficients + RMSE."""
    if len(times) < 4 or times[-1] - times[0] < 0.08:
        return None
    ts = [t - times[0] for t in times]
    sums = [sum(t**p for t in ts) for p in range(5)]
    normal = [[sums[i + j] for j in range(3)] for i in range(3)]
    inv = inverse(normal)
    if inv is None:
        return None
    c, b, a = multiply(inv, [sum(t**p * y for t, y in zip(ts, values)) for p in range(3)])
    error = math.sqrt(sum((y - (a * t * t + b * t + c)) ** 2 for t, y in zip(ts, values)) / len(ts))
    return a, b, c, error
