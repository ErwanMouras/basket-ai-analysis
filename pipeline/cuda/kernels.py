"""Fused CUDA kernels: no host arithmetic or data-dependent host loops.

Triton compiles these functions for the installed GPU. The small sequential
parts of LAP and convex hull construction run inside the GPU kernel as well.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def _cuda(*values):
    if not all(value.is_cuda for value in values):
        raise RuntimeError("GPU-only violation: native kernel requires CUDA tensors")


@triton.jit(do_not_specialize=["N", "M", "S0", "S1"])
def _assignment(
    C,
    U,
    V,
    P,
    OUT,
    N,
    M,
    S0,
    S1,
    LIMIT: tl.constexpr,
    B: tl.constexpr,
):
    col = tl.arange(0, B)
    cols = M + N
    failed = tl.full((), False, tl.int1)
    row = 1
    while (row <= N) & (~failed):
        tl.store(P, row)
        tl.debug_barrier()
        column = tl.full((), 0, tl.int32)
        minimum = tl.full((B,), float("inf"), tl.float64)
        used = tl.full((B,), False, tl.int1)
        way = tl.full((B,), 0, tl.int32)
        searching = tl.full((), True, tl.int1)
        steps = 0
        while searching & (steps <= N) & (~failed):
            steps += 1
            used = used | (col == column)
            active_row = tl.load(P + column)
            edge = tl.load(
                C + (active_row - 1) * S0 + (col - 1) * S1,
                (col > 0) & (col <= M),
                other=0,
            ).to(tl.float64)
            edge = tl.where(col > M, LIMIT, edge)
            reduced = (
                edge - tl.load(U + active_row) - tl.load(V + col, col <= cols, other=0)
            )
            better = (~used) & (col > 0) & (col <= cols) & (reduced < minimum)
            minimum = tl.where(better, reduced, minimum)
            way = tl.where(better, column, way)
            candidates = tl.where(
                (~used) & (col > 0) & (col <= cols), minimum, float("inf")
            )
            delta = tl.min(candidates, 0)
            eligible = (~used) & (col > 0) & (col <= cols)
            next_column = tl.min(tl.where(eligible & (candidates == delta), col, B), 0)
            finite = (delta < float("inf")) & (delta > -float("inf")) & (next_column <= cols)
            failed = ~finite
            owner = tl.load(P + col, col <= cols, other=0)
            previous_u = tl.load(U + owner, used & (col <= cols), other=0)
            tl.store(U + owner, previous_u + delta, used & (col <= cols) & finite)
            previous_v = tl.load(V + col, col <= cols, other=0)
            tl.store(V + col, previous_v - delta, used & (col <= cols) & finite)
            tl.debug_barrier()
            minimum = tl.where(~used, minimum - delta, minimum)
            column = tl.where(finite, next_column, 0)
            searching = tl.load(P + column) != 0
        failed |= searching
        steps = 0
        while (column != 0) & (steps <= N) & (~failed):
            steps += 1
            previous = tl.sum(tl.where(col == column, way, 0), 0)
            tl.store(P + column, tl.load(P + previous))
            tl.debug_barrier()
            column = previous
        failed |= column != 0
        row += 1
    owner = tl.load(P + col, (col > 0) & (col <= M), other=0) - 1
    cost = tl.load(
        C + owner * S0 + (col - 1) * S1,
        (owner >= 0) & (col > 0) & (col <= M),
        other=float("inf"),
    )
    tl.store(
        OUT + col - 1,
        tl.where((owner >= 0) & (cost <= LIMIT), owner, -1),
        (col > 0) & (col <= M),
    )
    tl.store(OUT + M, failed.to(tl.int32))


def assignment(cost, threshold):
    _cuda(cost)
    n, m = cost.shape
    u = torch.zeros(n + 1, device=cost.device, dtype=torch.float64)
    v = torch.zeros(n + m + 1, device=cost.device, dtype=torch.float64)
    p = torch.zeros(n + m + 1, device=cost.device, dtype=torch.int32)
    result = torch.empty(m + 1, device=cost.device, dtype=torch.int32)
    _assignment[(1,)](
        cost,
        u,
        v,
        p,
        result,
        n,
        m,
        *cost.stride(),
        threshold,
        triton.next_power_of_2(n + m + 1),
        num_warps=4,
    )
    if bool(result[m]):
        raise RuntimeError("CUDA assignment failed: invalid or cyclic augmenting path")
    return result[:m]


@triton.jit(do_not_specialize=["N"])
def _hull(POINTS, ORDER, STACK, SIZE, N, S0: tl.constexpr, S1: tl.constexpr):
    length = 0
    # Lower and upper monotone chains, with the same strict cross-product test.
    for half in range(2):
        base = length
        for offset in range(N):
            position = tl.where(half == 0, offset, N - 1 - offset)
            index = tl.load(ORDER + position)
            x = tl.load(POINTS + index * S0)
            y = tl.load(POINTS + index * S0 + S1)
            searching = length - base >= 2
            while searching:
                ia = tl.load(STACK + length - 2)
                ib = tl.load(STACK + length - 1)
                ax = tl.load(POINTS + ia * S0)
                ay = tl.load(POINTS + ia * S0 + S1)
                bx = tl.load(POINTS + ib * S0)
                by = tl.load(POINTS + ib * S0 + S1)
                cross = (bx - ax) * (y - by) - (by - ay) * (x - bx)
                if cross > 0:
                    searching = False
                else:
                    length -= 1
                    searching = length - base >= 2
            tl.store(STACK + length, index)
            length += 1
        length -= 1
    tl.store(SIZE, length)


def convex_hull(points):
    _cuda(points)
    if len(points) < 2:
        return points
    order = points[:, 1].argsort(stable=True)
    order = order[points[order, 0].argsort(stable=True)]
    stack = torch.empty(2 * len(points), device=points.device, dtype=torch.int64)
    length = torch.empty((), device=points.device, dtype=torch.int32)
    _hull[(1,)](
        points,
        order,
        stack,
        length,
        len(points),
        *points.stride(),
        num_warps=1,
        enable_fp_fusion=False,
    )
    # One host export for output shape, instead of synchronizing each hull edge.
    return points[stack[: int(length)]]


@triton.jit
def _bilinear(IMAGE, x, y, H: tl.constexpr, W: tl.constexpr):
    # grid_sample reflection, align_corners=False: reflect around pixel edges.
    rx = tl.abs(x + 0.5)
    ry = tl.abs(y + 0.5)
    qx = tl.floor(rx / W)
    qy = tl.floor(ry / H)
    rx = rx - qx * W
    ry = ry - qy * H
    x = tl.minimum(
        tl.maximum(tl.where(qx.to(tl.int32) % 2 == 0, rx, W - rx) - 0.5, 0), W - 1
    )
    y = tl.minimum(
        tl.maximum(tl.where(qy.to(tl.int32) % 2 == 0, ry, H - ry) - 0.5, 0), H - 1
    )
    x0 = tl.floor(x).to(tl.int32)
    y0 = tl.floor(y).to(tl.int32)
    x1 = tl.minimum(x0 + 1, W - 1)
    y1 = tl.minimum(y0 + 1, H - 1)
    fx, fy = x - x0, y - y0
    a = tl.load(IMAGE + y0 * W + x0)
    b = tl.load(IMAGE + y0 * W + x1)
    c = tl.load(IMAGE + y1 * W + x0)
    d = tl.load(IMAGE + y1 * W + x1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


@triton.jit
def _lk_level(
    SOURCE,
    TARGET,
    GX,
    GY,
    POINTS,
    FOLLOWING,
    VALID,
    OUT,
    H: tl.constexpr,
    W: tl.constexpr,
    SCALE: tl.constexpr,
    INITIAL: tl.constexpr,
    B: tl.constexpr,
):
    point = tl.program_id(0)
    lane = tl.arange(0, B)
    keep = lane < 441
    px = (lane % 21).to(tl.float32) - 10
    py = (lane // 21).to(tl.float32) - 10
    anchor_x = tl.load(POINTS + point * 2) / SCALE
    anchor_y = tl.load(POINTS + point * 2 + 1) / SCALE
    if INITIAL:
        x, y = anchor_x, anchor_y
    else:
        x = tl.load(FOLLOWING + point * 2) * 2
        y = tl.load(FOLLOWING + point * 2 + 1) * 2
    ref = _bilinear(SOURCE, anchor_x + px, anchor_y + py, H, W)
    gx = _bilinear(GX, anchor_x + px, anchor_y + py, H, W)
    gy = _bilinear(GY, anchor_x + px, anchor_y + py, H, W)
    a = tl.sum(tl.where(keep, gx * gx, 0), 0)
    b = tl.sum(tl.where(keep, gx * gy, 0), 0)
    c = tl.sum(tl.where(keep, gy * gy, 0), 0)
    determinant = a * c - b * b
    valid = tl.load(VALID + point) & (determinant > 1.0e-4)
    safe = tl.maximum(determinant, 1.0e-4)
    active = valid
    iteration = 0
    while active & (iteration < 30):
        error = ref - _bilinear(TARGET, x + px, y + py, H, W)
        bx = tl.sum(tl.where(keep, gx * error, 0), 0)
        by = tl.sum(tl.where(keep, gy * error, 0), 0)
        dx = tl.minimum(tl.maximum((c * bx - b * by) / safe, -5), 5)
        dy = tl.minimum(tl.maximum((a * by - b * bx) / safe, -5), 5)
        current_cost = tl.sum(tl.where(keep, error * error, 0), 0)
        best_cost = float("inf")
        best_step = 0.0
        for trial in tl.static_range(4):
            step = 0.5**trial
            residual = ref - _bilinear(
                TARGET, x + step * dx + px, y + step * dy + py, H, W
            )
            cost = tl.sum(tl.where(keep, residual * residual, 0), 0)
            better = cost < best_cost
            best_step = tl.where(better, step, best_step)
            best_cost = tl.minimum(best_cost, cost)
        improved = best_cost < current_cost
        x = tl.where(improved, x + best_step * dx, x)
        y = tl.where(improved, y + best_step * dy, y)
        active = improved & (tl.sqrt(dx * dx + dy * dy) * best_step >= 0.01)
        iteration += 1
    valid = valid & (tl.abs(x) < float("inf")) & (tl.abs(y) < float("inf"))
    tl.store(OUT + point * 2, x)
    tl.store(OUT + point * 2 + 1, y)
    tl.store(VALID + point, valid)


def lk_level(source, target, gradients, points, following, valid, level):
    _cuda(source, target, gradients, points, following, valid)
    result = torch.empty_like(points)
    _lk_level[(len(points),)](
        source,
        target,
        gradients[0],
        gradients[1],
        points,
        following,
        valid,
        result,
        *source.shape,
        2**level,
        level == 3,
        512,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


@triton.jit
def _floor_mask(
    POLYGON,
    ROWS,
    REGIONS,
    OUT,
    N,
    NR,
    RS: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    HAS_POLYGON: tl.constexpr,
    B: tl.constexpr,
):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    x = (lane % W).to(tl.float64) * (IW / W)
    y = (lane // W).to(tl.float64) * (IH / H)
    if HAS_POLYGON:
        positive = tl.full((B,), True, tl.int1)
        negative = tl.full((B,), True, tl.int1)
        for edge in tl.static_range(4):
            ax = tl.load(POLYGON + edge * 2)
            ay = tl.load(POLYGON + edge * 2 + 1)
            bx = tl.load(POLYGON + ((edge + 1) % 4) * 2)
            by = tl.load(POLYGON + ((edge + 1) % 4) * 2 + 1)
            cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
            positive &= cross >= 0
            negative &= cross <= 0
        included = positive | negative
    else:
        included = y >= IH * 0.25
    for index in range(N):
        confidence = tl.load(ROWS + index * RS + 4)
        left = tl.load(ROWS + index * RS) - 8
        top = tl.load(ROWS + index * RS + 1) - 8
        right = tl.load(ROWS + index * RS + 2) + 8
        bottom = tl.load(ROWS + index * RS + 3) + 8
        included &= ~(
            (confidence >= 0.15)
            & (x >= left)
            & (x <= right)
            & (y >= top)
            & (y <= bottom)
        )
    for index in range(NR):
        left = tl.load(REGIONS + index * 4) * IW
        top = tl.load(REGIONS + index * 4 + 1) * IH
        right = tl.load(REGIONS + index * 4 + 2) * IW
        bottom = tl.load(REGIONS + index * 4 + 3) * IH
        included &= ~((x >= left) & (x <= right) & (y >= top) & (y <= bottom))
    tl.store(OUT + lane, included, lane < H * W)


def floor_mask(polygon, rows, regions, shape, image_shape):
    _cuda(rows, regions)
    if polygon is not None:
        _cuda(polygon)
    h, w = shape
    result = torch.empty((h, w), device=rows.device, dtype=torch.bool)
    _floor_mask[(triton.cdiv(h * w, 256),)](
        polygon if polygon is not None else rows,
        rows,
        regions,
        result,
        len(rows),
        len(regions),
        rows.stride(0),
        h,
        w,
        *image_shape,
        polygon is not None,
        256,
        enable_fp_fusion=False,
    )
    return result


@triton.jit(do_not_specialize=["CW", "CH", "SX", "SY"])
def _torso_color(
    IMAGE,
    MASK,
    OUT,
    CW,
    CH,
    SX,
    SY,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    MS0: tl.constexpr,
    MS1: tl.constexpr,
    HAS_MASK: tl.constexpr,
    B: tl.constexpr,
):
    lane = tl.arange(0, B)
    x, y = lane % CW + SX, lane // CW + SY
    keep = lane < CW * CH
    if HAS_MASK:
        keep &= tl.load(MASK + y * MS0 + x * MS1, lane < CW * CH, other=0)
    count = tl.sum(keep.to(tl.int32), 0)
    offset = y * S1 + x * S2
    r = tl.load(IMAGE + offset, lane < CW * CH, other=0).to(tl.float32) / 255
    g = tl.load(IMAGE + S0 + offset, lane < CW * CH, other=0).to(tl.float32) / 255
    b = tl.load(IMAGE + 2 * S0 + offset, lane < CW * CH, other=0).to(tl.float32) / 255
    maximum = tl.maximum(tl.maximum(r, g), b)
    minimum = tl.minimum(tl.minimum(r, g), b)
    delta = maximum - minimum
    saturation = delta / tl.where(maximum == 0, 1, maximum + 1.0e-8) * 255
    divisor = tl.where(delta == 0, 1, delta)
    rc, gc, bc = maximum - r, maximum - g, maximum - b
    hue = (
        tl.where(
            (r >= g) & (r >= b),
            bc - gc,
            tl.where(g >= b, (rc - bc) + 2 * divisor, (gc - rc) + 4 * divisor),
        )
        / divisor
    )
    hue = hue / 6
    hue = (hue - tl.floor(hue)) * (2 * 3.141592653589793)
    hue = tl.where(hue >= 2 * 3.141592653589793, 0, hue)
    hue = hue * (180 / (2 * 3.141592653589793))
    value = maximum * 255
    colored = keep & (saturation >= 75) & (value >= 50)
    colored_count = tl.sum(colored.to(tl.int32), 0)
    bins = tl.minimum(tl.maximum(tl.floor(hue / 10).to(tl.int32), 0), 17)
    histogram = tl.histogram(tl.where(colored, bins, 31), 32)
    entries = tl.arange(0, 32)
    peak = tl.max(tl.where(entries < 18, histogram, -1), 0)
    centre = tl.min(tl.where((entries < 18) & (histogram == peak), entries, 32), 0)
    relative = tl.floor(hue / 10) - centre + 1
    relative = relative - tl.floor(relative / 18) * 18
    inliers = colored & (relative <= 2)
    inlier_count = tl.sum(inliers.to(tl.int32), 0)
    angle = hue * (2 * 3.141592653589793 / 180)
    sine = tl.sum(tl.where(inliers, tl.sin(angle), 0), 0) / tl.maximum(inlier_count, 1)
    cosine = tl.sum(tl.where(inliers, tl.cos(angle), 0), 0) / tl.maximum(
        inlier_count, 1
    )
    result = libdevice.atan2(sine, cosine)
    result = result - tl.floor(result / (2 * 3.141592653589793)) * (
        2 * 3.141592653589793
    )
    result = result * 180 / (2 * 3.141592653589793)
    dark = (
        tl.sum((keep & (value <= 85)).to(tl.int32), 0).to(tl.float32)
        / tl.maximum(count, 1)
        >= 0.7
    )
    light = (
        tl.sum((keep & (value >= 175) & (saturation <= 60)).to(tl.int32), 0).to(
            tl.float32
        )
        / tl.maximum(count, 1)
        >= 0.7
    )
    hue_valid = (colored_count.to(tl.float32) / tl.maximum(count, 1) >= 0.7) & (
        inlier_count.to(tl.float32) / tl.maximum(colored_count, 1) >= 0.8
    )
    kind = tl.where(hue_valid, 1, tl.where(dark, 2, tl.where(light, 3, 0)))
    if HAS_MASK:
        kind = tl.where(
            (count >= 16) & (count.to(tl.float32) / (CW * CH) >= 0.5), kind, 0
        )
    tl.store(OUT, kind.to(tl.float32))
    tl.store(OUT + 1, tl.where(kind == 1, result, 0))


def torso_color(image, foreground):
    _cuda(image)
    if foreground is not None:
        _cuda(foreground)
    h, w = image.shape[-2:]
    sy, sx = h // 10, w // 10
    cw, ch = w - 2 * sx, h - 2 * sy
    result = torch.empty(2, device=image.device, dtype=torch.float32)
    strides = foreground.stride() if foreground is not None else (0, 0)
    _torso_color[(1,)](
        image,
        foreground if foreground is not None else image,
        result,
        cw,
        ch,
        sx,
        sy,
        *image.stride(),
        *strides,
        foreground is not None,
        triton.next_power_of_2(cw * ch),
        num_warps=8,
        enable_fp_fusion=False,
    )
    return result


@triton.jit(do_not_specialize=["N"])
def _episodes(CODES, STABLE, GAPS, NEXT, STARTS, N):
    current = tl.load(NEXT)
    tl.store(STARTS, 1)
    for index in range(1, N):
        gap = tl.load(GAPS + index - 1)
        code = tl.load(CODES + index)
        stable = tl.load(STABLE + index)
        switch = (code >= 0) & (current >= 0) & (code != current) & stable
        boundary = gap | switch
        tl.store(STARTS + index, boundary)
        current = tl.where(
            gap,
            tl.load(NEXT + index),
            tl.where(switch | ((code >= 0) & (current < 0)), code, current),
        )


def episode_boundaries(codes, stable, gaps):
    _cuda(codes, stable, gaps)
    n = len(codes)
    indices = torch.arange(n, device=codes.device)
    next_index = torch.where(codes >= 0, indices, n).flip(0).cummin(0).values.flip(0)
    next_code = torch.where(next_index < n, codes[next_index.clamp_max(n - 1)], -1)
    starts = torch.empty(n, device=codes.device, dtype=torch.bool)
    _episodes[(1,)](codes, stable, gaps, next_code, starts, n, num_warps=1)
    return torch.nonzero(starts).flatten().tolist()


@triton.jit
def _quad(x0, y0, x1, y1, x2, y2, x3, y3):
    dx1, dx2 = x1 - x2, x3 - x2
    dy1, dy2 = y1 - y2, y3 - y2
    dx3, dy3 = x0 - x1 + x2 - x3, y0 - y1 + y2 - y3
    denominator = dx1 * dy2 - dx2 * dy1
    g = (dx3 * dy2 - dx2 * dy3) / denominator
    h = (dx1 * dy3 - dx3 * dy1) / denominator
    return (
        x1 - x0 + g * x1,
        x3 - x0 + h * x3,
        x0,
        y1 - y0 + g * y1,
        y3 - y0 + h * y3,
        y0,
        g,
        h,
        tl.full(x0.shape, 1.0, tl.float64),
    )


@triton.jit
def _inverse3(a):
    c0 = a[4] * a[8] - a[5] * a[7]
    c1 = a[2] * a[7] - a[1] * a[8]
    c2 = a[1] * a[5] - a[2] * a[4]
    c3 = a[5] * a[6] - a[3] * a[8]
    c4 = a[0] * a[8] - a[2] * a[6]
    c5 = a[2] * a[3] - a[0] * a[5]
    c6 = a[3] * a[7] - a[4] * a[6]
    c7 = a[1] * a[6] - a[0] * a[7]
    c8 = a[0] * a[4] - a[1] * a[3]
    determinant = a[0] * c0 + a[1] * c3 + a[2] * c6
    return (
        c0 / determinant,
        c1 / determinant,
        c2 / determinant,
        c3 / determinant,
        c4 / determinant,
        c5 / determinant,
        c6 / determinant,
        c7 / determinant,
        c8 / determinant,
    )


@triton.jit
def _mm3(a, b):
    return (
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    )


@triton.jit
def _normalized_quad(POINTS, index, valid):
    x0 = tl.load(POINTS + index * 8, valid, other=0)
    y0 = tl.load(POINTS + index * 8 + 1, valid, other=0)
    x1 = tl.load(POINTS + index * 8 + 2, valid, other=0)
    y1 = tl.load(POINTS + index * 8 + 3, valid, other=0)
    x2 = tl.load(POINTS + index * 8 + 4, valid, other=0)
    y2 = tl.load(POINTS + index * 8 + 5, valid, other=0)
    x3 = tl.load(POINTS + index * 8 + 6, valid, other=0)
    y3 = tl.load(POINTS + index * 8 + 7, valid, other=0)
    cx, cy = (x0 + x1 + x2 + x3) / 4, (y0 + y1 + y2 + y3) / 4
    x0, x1, x2, x3 = x0 - cx, x1 - cx, x2 - cx, x3 - cx
    y0, y1, y2, y3 = y0 - cy, y1 - cy, y2 - cy, y3 - cy
    mean = (
        tl.sqrt(x0 * x0 + y0 * y0)
        + tl.sqrt(x1 * x1 + y1 * y1)
        + tl.sqrt(x2 * x2 + y2 * y2)
        + tl.sqrt(x3 * x3 + y3 * y3)
    ) / 4
    scale = 1.4142135623730951 / tl.maximum(mean, 1.0e-12)
    q = _quad(
        x0 * scale,
        y0 * scale,
        x1 * scale,
        y1 * scale,
        x2 * scale,
        y2 * scale,
        x3 * scale,
        y3 * scale,
    )
    return q, cx, cy, scale


@triton.jit
def _four_point(FIRST, SECOND, OUT, N, B: tl.constexpr):
    index = tl.program_id(0) * B + tl.arange(0, B)
    valid = index < N
    qa, ax, ay, sa = _normalized_quad(FIRST, index, valid)
    qb, bx, by, sb = _normalized_quad(SECOND, index, valid)
    h = _mm3(qb, _inverse3(qa))
    a0, a1, a2 = h[0] * sa, h[1] * sa, h[2] - h[0] * sa * ax - h[1] * sa * ay
    a3, a4, a5 = h[3] * sa, h[4] * sa, h[5] - h[3] * sa * ax - h[4] * sa * ay
    a6, a7, a8 = h[6] * sa, h[7] * sa, h[8] - h[6] * sa * ax - h[7] * sa * ay
    result = (
        a0 / sb + bx * a6,
        a1 / sb + bx * a7,
        a2 / sb + bx * a8,
        a3 / sb + by * a6,
        a4 / sb + by * a7,
        a5 / sb + by * a8,
        a6,
        a7,
        a8,
    )
    denominator = tl.where(tl.abs(a8) > 1.0e-12, a8, 1.0)
    for entry in tl.static_range(9):
        tl.store(OUT + index * 9 + entry, result[entry] / denominator, valid)


def four_point(first, second):
    _cuda(first, second)
    first, second = first.double().contiguous(), second.double().contiguous()
    out = torch.empty((len(first), 3, 3), device=first.device, dtype=torch.float64)
    _four_point[(triton.cdiv(len(first), 128),)](
        first, second, out, len(first), 128, num_warps=4, enable_fp_fusion=False
    )
    return out


@triton.jit(do_not_specialize=["FLAGS"])
def _metric_step(
    POINT,
    PREVIOUS,
    FILTERED,
    ANCHOR,
    DT,
    PARAMS,
    DISTANCE,
    TRACKED,
    MEASURED,
    PAIRS,
    OUT,
    FLAGS,
):
    dt = tl.load(DT)
    max_gap = tl.load(PARAMS)
    max_speed = tl.load(PARAMS + 1)
    tau = tl.load(PARAMS + 2)
    deadband = tl.load(PARAMS + 3)
    same = (FLAGS & 1) != 0
    conflict = (FLAGS & 2) != 0
    missing = (FLAGS & 4) != 0
    method_changed = (FLAGS & 8) != 0
    x, y = tl.load(POINT), tl.load(POINT + 1)
    dx, dy = x - tl.load(PREVIOUS), y - tl.load(PREVIOUS + 1)
    speed = tl.sqrt(dx * dx + dy * dy) / dt
    reason = tl.where(
        ~same | conflict,
        1,
        tl.where(
            dt > max_gap,
            2,
            tl.where(
                missing,
                3,
                tl.where(method_changed, 4, tl.where(speed > max_speed, 5, 0)),
            ),
        ),
    )
    alpha = tl.where(tau != 0, 1 - libdevice.exp(-dt / tau), 1.0)
    fx = tl.load(FILTERED) + alpha * (x - tl.load(FILTERED))
    fy = tl.load(FILTERED + 1) + alpha * (y - tl.load(FILTERED + 1))
    ax, ay = tl.load(ANCHOR), tl.load(ANCHOR + 1)
    sx, sy = fx - ax, fy - ay
    step = tl.sqrt(sx * sx + sy * sy)
    distance = tl.where((reason == 0) & (step >= deadband), step, 0.0)
    tl.store(OUT, tl.where(reason == 0, fx, x))
    tl.store(OUT + 1, tl.where(reason == 0, fy, y))
    tl.store(OUT + 2, tl.where(reason == 0, tl.where(distance != 0, fx, ax), x))
    tl.store(OUT + 3, tl.where(reason == 0, tl.where(distance != 0, fy, ay), y))
    tl.store(OUT + 4, distance)
    tl.store(OUT + 5, reason.to(tl.float64))
    tl.store(DISTANCE, tl.load(DISTANCE) + distance)
    tl.store(TRACKED, tl.load(TRACKED) + tl.where(same, dt, 0.0))
    tl.store(MEASURED, tl.load(MEASURED) + tl.where(reason == 0, dt, 0.0))
    tl.store(PAIRS, tl.load(PAIRS) + tl.where(reason == 0, 1.0, 0.0))


def metric_step(point, previous, filtered, anchor, dt, params, row, flags):
    totals = [
        row[key]
        for key in (
            "distance",
            "tracked_duration_s",
            "measured_duration_s",
            "valid_pairs",
        )
    ]
    _cuda(point, previous, filtered, anchor, dt, params, *totals)
    out = torch.empty(6, device=point.device, dtype=torch.float64)
    _metric_step[(1,)](
        point,
        previous,
        filtered,
        anchor,
        dt,
        params,
        *totals,
        out,
        flags,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out
