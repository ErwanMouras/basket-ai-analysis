const test = require("node:test");
const assert = require("node:assert/strict");
const { density, subtract, maximum, sample } = require("./heatmap-renderer.js");
const integral = (field) =>
  field.values.reduce((sum, value) => sum + value, 0) * field.area;
const close = (actual, expected, epsilon = 1e-9) =>
  assert.ok(Math.abs(actual - expected) < epsilon, `${actual} ≠ ${expected}`);

// A color reconstruction must not erase contributions near the boundaries.
test("density preserves mass at corners, sidelines and inside the court", () => {
  for (const index of [0, 8, 40, 80]) {
    const cells = Array(81).fill(0);
    cells[index] = 100;
    const field = density(cells, 9, 9, 9, 9);
    close(integral(field), 100);
    assert.ok(
      field.values.every((value) => Number.isFinite(value) && value >= 0),
    );
    assert.equal(cells[index], 100, "raw measurements remain unchanged");
  }
});

test("nearby observations form a continuous field instead of colored squares", () => {
  const field = density([0, 0, 0, 0, 100, 0, 0, 0, 0], 3, 3, 9, 9);
  const a = sample(field, 0.5, 0.5);
  const b = sample(field, 0.55, 0.5);
  const c = sample(field, 0.6, 0.5);
  assert.ok(
    a > b && b > c && c > 0,
    "concentration falls smoothly away from an observation",
  );
  close(sample(field, 0.55, 0.5), sample(field, 0.5, 0.55));
});

test("the same physical distance has the same spread on a rectangular court", () => {
  const values = Array(45).fill(0);
  values[22] = 100;
  const field = density(values, 9, 5, 27, 15);
  close(sample(field, 0.5 + 1 / 27, 0.5), sample(field, 0.5, 0.5 + 1 / 15));
});

test("comparison uses linear densities with balanced signed differences", () => {
  const a = density([80, 20, 0, 0], 2, 2, 10, 6);
  const b = density([0, 0, 20, 80], 2, 2, 10, 6);
  const difference = subtract(a, b);
  const inverse = subtract(b, a);
  close(integral(difference), 0);
  assert.ok(difference.values.some((value) => value > 0));
  assert.ok(difference.values.some((value) => value < 0));
  difference.values.forEach((value, index) =>
    close(value, -inverse.values[index]),
  );
  close(maximum(difference), maximum(inverse));
  close(maximum(subtract(a, a)), 0);
});

test("missing profiles stay neutral and absolute scale remains comparable", () => {
  const a = density([100, 0, 0, 0], 2, 2, 10, 6);
  const half = density([50, 0, 0, 0], 2, 2, 10, 6);
  const empty = density([0, 0, 0, 0], 2, 2, 10, 6);
  close(maximum(a), 2 * maximum(half));
  close(maximum(empty), 0);
  close(maximum(subtract(a, empty, false)), 0);
});

test("NBA half-court density keeps 100% at midcourt and baseline boundaries", () => {
  const length = 28.6512 / 2,
    width = 15.24;
  const nx = Math.ceil(length),
    ny = Math.ceil(width);
  for (const index of [0, nx - 1, nx * (ny - 1), nx * ny - 1]) {
    const cells = Array(nx * ny).fill(0);
    cells[index] = 100;
    const field = density(cells, nx, ny, length, width);
    close(integral(field), 100);
    assert.ok(
      field.values.every((value) => Number.isFinite(value) && value >= 0),
    );
    assert.ok(
      field.rows > field.cols,
      "half-court dimensions retain physical scale",
    );
  }
});
