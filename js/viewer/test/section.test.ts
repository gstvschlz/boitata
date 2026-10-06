import { describe, expect, it } from "vitest";
import type { Vec3 } from "../src/bounds";
import { barHeights } from "../src/filter";
import { buildSection, inSlab, pickCut, type Section, segmentOf, snap45, traces, unfold } from "../src/section";

const section = (points: [number, number][], width = 10, dip = 90, z = 0): Section => buildSection(points, z, width, dip)!;
const close = (a: number[], b: number[], digits = 9) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], digits));

describe("45 degree lock", () => {
  it("moves the point onto the nearest multiple of 45 degrees from north, keeping its projection", () => {
    close(snap45([0, 0], [10, 1]), [10, 0]);
    close(snap45([0, 0], [3, 2.9]), [5.9 / 2, 5.9 / 2]);
    close(snap45([5, 5], [4, -5]), [5, -5]);
    close(snap45([0, 0], [-7, 7.2]), [-7.1, 7.1]);
  });
});

describe("slab membership", () => {
  const straight = section([
    [0, 0],
    [100, 0],
  ]);

  it("keeps points within half the width of an upright section, between its ends", () => {
    expect(inSlab(straight, [50, 4.9, 700])).toBe(true);
    expect(inSlab(straight, [50, -4.9, -700])).toBe(true);
    expect(inSlab(straight, [50, 5.1, 0])).toBe(false);
    expect(inSlab(straight, [-0.1, 0, 0])).toBe(false);
    expect(inSlab(straight, [100.1, 0, 0])).toBe(false);
  });

  it("tilts a dipping section down to the right of the direction it is drawn in", () => {
    const dipping = section(
      [
        [0, 0],
        [100, 0],
      ],
      10,
      45,
    );
    expect(inSlab(dipping, [50, -40, -40])).toBe(true);
    expect(inSlab(dipping, [50, 40, 40])).toBe(true);
    expect(inSlab(dipping, [50, 0, -10])).toBe(false);
    expect(inSlab(dipping, [50, -3, 3])).toBe(true);
    expect(inSlab(dipping, [50, 40, -40])).toBe(false);
  });

  it("joins the segments of a polyline on the bisectors of its bends", () => {
    const bent = section([
      [0, 0],
      [100, 0],
      [100, 100],
    ]);
    expect(inSlab(bent, [50, -4, 0])).toBe(true);
    expect(inSlab(bent, [104, 50, 0])).toBe(true);
    expect(inSlab(bent, [104, -4, 0])).toBe(true);
    expect(inSlab(bent, [96, 4, 0])).toBe(true);
    expect(inSlab(bent, [106, -6, 0])).toBe(false);
    expect(inSlab(bent, [50, 50, 0])).toBe(false);
    expect(inSlab(bent, [100, 100.1, 0])).toBe(false);
  });

  it("drops repeated vertices and needs two distinct ones", () => {
    expect(buildSection([[1, 1], [1, 1]], 0, 10, 90)).toBeNull();
    expect(section([[0, 0], [0, 0], [10, 0]]).segments).toHaveLength(1);
  });
});

describe("unfolding", () => {
  const zigzag = section([
    [0, 0],
    [30, 40],
    [90, 40],
    [90, -10],
  ]);

  it("measures the distance along the section continuously across its vertices", () => {
    const vertices: [number, number][] = [
      [30, 40],
      [90, 40],
    ];
    const at = [50, 110];
    vertices.forEach(([x, y], k) => {
      for (const ref of [
        [x - 0.01 * (k === 0 ? 30 : 60), y - (k === 0 ? 0.4 : 0), 0],
        [x + (k === 0 ? 0.6 : 0), y - (k === 0 ? 0 : 0.5), 0],
      ] as Vec3[])
        close(unfold(zigzag, [x, y, 7], ref), [at[k], 0, 7], 6);
    });
    expect(zigzag.length).toBeCloseTo(160);
  });

  it("lays each segment out along x, its left side toward +y, elevation kept", () => {
    close(unfold(zigzag, [15, 20, -3]), [25, 0, -3]);
    close(unfold(zigzag, [60, 45, 1]), [80, 5, 1]);
    close(unfold(zigzag, [85, 0, 2]), [150, -5, 2]);
    const xs = [0, 0.25, 0.5, 0.75, 1].map((t) => unfold(zigzag, [30 + 60 * t, 40, 0])[0]);
    expect(xs).toEqual([...xs].sort((a, b) => a - b));
  });

  it("assigns points to the segment whose stretch holds them", () => {
    expect(segmentOf(zigzag, [-50, -50, 0])).toBe(0);
    expect(segmentOf(zigzag, [60, 60, 0])).toBe(1);
    expect(segmentOf(zigzag, [200, -100, 0])).toBe(2);
  });
});

describe("outline traces", () => {
  it("offsets the slab edges by half the width, mitered at bends", () => {
    const [trace, left, right] = traces(
      section([
        [0, 0],
        [100, 0],
        [100, 100],
      ]),
      0,
    );
    close(trace.flat(), [0, 0, 100, 0, 100, 100]);
    close(left.flat(), [0, 5, 95, 5, 95, 100]);
    close(right.flat(), [0, -5, 105, -5, 105, 100]);
  });

  it("follows a dipping section's surface up and down", () => {
    const dipping = section(
      [
        [0, 0],
        [100, 0],
      ],
      10,
      45,
    );
    close(traces(dipping, 20)[0].flat(), [0, 20, 100, 20]);
    close(traces(dipping, -20)[0].flat(), [0, -20, 100, -20]);
  });
});

describe("quick cut picking", () => {
  it("picks on the level plane through the center when looking down", () => {
    const forward: Vec3 = [0, Math.cos(Math.PI / 4), -Math.sin(Math.PI / 4)];
    const rays = [
      { origin: [-10, -100, 100] as Vec3, direction: forward },
      { origin: [30, -100, 100] as Vec3, direction: forward },
    ] as const;
    const ends = pickCut([rays[0], rays[1]], [0, 0, 20], forward)!;
    close(ends[0], [-10, -20, 20]);
    close(ends[1], [30, -20, 20]);
    const cut = section([
      [ends[0][0], ends[0][1]],
      [ends[1][0], ends[1][1]],
    ]);
    expect(inSlab(cut, [10, -20, -500])).toBe(true);
    expect(inSlab(cut, [10, 0, 0])).toBe(false);
  });

  it("picks on the upright plane facing the camera in a level view", () => {
    const forward: Vec3 = [1, 0, 0];
    const ends = pickCut(
      [
        { origin: [-100, 10, 0], direction: [1, 0, 0] },
        { origin: [-100, -10, 5], direction: [1, 0, 0] },
      ],
      [7, 0, 0],
      forward,
    )!;
    close(ends[0], [7, 10, 0]);
    close(ends[1], [7, -10, 5]);
  });

  it("returns null when a ray misses its plane", () => {
    const forward: Vec3 = [0, 0, -1];
    expect(
      pickCut(
        [
          { origin: [0, 0, 10], direction: [0, 0, -1] },
          { origin: [0, 0, 10], direction: [0, 0, 1] },
        ],
        [0, 0, 0],
        forward,
      ),
    ).toBeNull();
  });
});

describe("histogram bars", () => {
  it("keeps a tail of a few rows visible beside a spike", () => {
    const heights = barHeights([10_000, 0, 1, 3, 40]);
    expect(heights[0]).toBe(1);
    expect(heights[1]).toBe(0);
    expect(heights[2]).toBeGreaterThan(0.07);
    expect(heights[2]).toBeLessThan(heights[3]);
    expect(heights[3]).toBeLessThan(heights[4]);
    expect(Math.sqrt(1 / 10_000)).toBeLessThan(0.06);
  });

  it("draws nothing for an empty histogram", () => {
    expect(barHeights([0, 0])).toEqual([0, 0]);
  });
});
