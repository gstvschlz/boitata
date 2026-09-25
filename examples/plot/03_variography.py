import matplotlib.pyplot as plt
import numpy as np

from style import ACCENT, GREY, HIGHLIGHT, INK, OUT, save, table

plane = table(OUT / "03_plane.csv")
m = {k: v[0] for k, v in table(OUT / "03_model.csv").items()}
angles = np.unique(plane["angle"])
lags = np.unique(plane["lag"])
gamma = plane["gamma"].reshape(angles.size, lags.size)
count = plane["count"].reshape(angles.size, lags.size)
gamma = np.where(count >= 30, gamma, np.nan)

step = angles[1] - angles[0]
theta = np.concatenate([angles, angles + np.pi, [2 * np.pi]]) - step / 2
width = lags[1] - lags[0]
radius = np.concatenate([lags - width / 2, [lags[-1] + width / 2]])
values = np.vstack([gamma, gamma])

fig = plt.figure(figsize=(9.2, 4.2), layout="constrained")
a = fig.add_subplot(1, 2, 1, projection="polar")
mesh = a.pcolormesh(theta, radius, values.T / m["variance"], vmin=0, vmax=1.2, shading="flat")
t = np.linspace(0, 2 * np.pi, 361)
major = np.deg2rad(90 - m["azimuth"])
ellipse = m["major"] * m["minor"] / np.hypot(m["minor"] * np.cos(t - major), m["major"] * np.sin(t - major))
a.plot(t, ellipse, color=HIGHLIGHT, lw=1.4)
a.set_rlim(0, radius[-1])
a.set_xticks(np.deg2rad([0, 90, 180, 270]), ["E", "N", "W", "S"])
a.set_rlabel_position(200)
a.tick_params(labelsize=7)
a.set_title("Variogram map (γ / variance)")
fig.colorbar(mesh, ax=a, shrink=0.7, label="γ / sample variance")
a.text(major, m["major"] * 1.05, f"{m['major']:.0f} m", color=HIGHLIGHT, fontsize=8)

b = fig.add_subplot(1, 2, 2)
for name, color, azimuth in (("major", ACCENT, m["azimuth"]), ("minor", GREY, m["azimuth"] + 90)):
    d = table(OUT / f"03_{name}.csv")
    keep = d["count"] > 0
    b.scatter(d["lag"][keep], d["gamma"][keep], s=np.sqrt(d["count"][keep]) * 2, color=color,
              label=f"N{azimuth % 360:.0f}° experimental")
    h = np.linspace(0, d["lag"].max(), 200)
    rng = m[name]
    shape = np.where(h < rng, 1.5 * h / rng - 0.5 * (h / rng) ** 3, 1.0)
    b.plot(h, m["nugget"] + m["sill"] * shape, color=color, lw=1.4)
    b.axvline(rng, color=color, lw=0.8, ls=":")
b.axhline(m["variance"], color=INK, lw=0.8, ls="--")
b.text(118, m["variance"], "sample variance", va="top", ha="right", color=INK, fontsize=8)
b.set_xlim(left=0)
b.set_ylim(bottom=0)
b.set_title("Directional variograms and fitted model")
b.set_xlabel("Lag distance (m)")
b.set_ylabel("γ(h) (ppm²)")
b.legend(loc="lower right", title="marker area ∝ pairs", title_fontsize=8)
save(fig, "03-variography", "variogram")
