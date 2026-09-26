import pickle

import ceres as cs
import matplotlib
import numpy as np
import pyarrow as pa
import pytest

matplotlib.use("Agg")

rng = np.random.default_rng(11)
POOL = np.array(["a", "b", "c", "d", "e", "7", "12", "100"])


def draws(k=30):
    for _ in range(k):
        n = rng.integers(1, 60)
        yield rng.choice(POOL[: rng.integers(1, POOL.size + 1)], n), rng.uniform(0, 10, n)


def test_round_trip_keeps_listed_labels_lumps_the_rest_and_nulls():
    for raw, _ in draws():
        names = sorted(set(raw))[: rng.integers(0, 4)]
        c = cs.Categories(names, other="other")
        values = [*raw, None, np.nan]
        expected = [v if v in names else "other" for v in raw] + [None, None]
        assert c.decode(c.encode(values)) == expected


def test_from_values_ignores_row_order():
    for raw, w in draws():
        min_share = rng.uniform(0, 0.3)
        a = cs.Categories.from_values(raw, weights=w, min_share=min_share)
        order = rng.permutation(raw.size)
        b = cs.Categories.from_values(raw[order], weights=w[order], min_share=min_share)
        assert a == b and a.to_json() == b.to_json()
        if a.other is not None:
            assert a.names[-1] == a.other


def test_integers_sort_numerically_and_integral_floats_are_integers():
    c = cs.Categories.from_values([10.0, 9, "100", np.nan, None, 9.0])
    assert c.names == ["9", "10", "100"] and c.other is None
    np.testing.assert_array_equal(c.encode(pa.array([9, None, 100])), [0, np.nan, 2])


def test_shares_sum_to_one_and_other_holds_the_lumped():
    for raw, w in draws():
        full = cs.Categories.from_values(raw, weights=w)
        full_shares = full.shares(full.encode(raw), weights=w)
        min_share = rng.uniform(0, 0.3)
        c = cs.Categories.from_values(raw, weights=w, min_share=min_share)
        shares = c.shares(c.encode(raw), weights=w)
        assert shares.sum() == pytest.approx(1.0)
        lumped = [n for n in full.names if n not in c.names]
        assert all(c.mapping[n] == "other" for n in lumped)
        if c.other is not None:
            summed = sum(s for n, s in zip(full.names, full_shares, strict=True) if n in lumped)
            assert shares[-1] == pytest.approx(summed)
        assert all(s >= min_share - 1e-12 for n, s in zip(c.names, shares, strict=True) if n != c.other)
        if lumped and len(lumped) < len(full):
            merged = full.lump(lumped)
            by_hand = ["other" if v in lumped else v for v in raw]
            np.testing.assert_array_equal(merged.encode(raw), merged.encode(by_hand))


def test_lump_and_mapping():
    c = cs.Categories(["MS", "SM", "RH"], colors=["r", "g", "b"], mapping={"MSX": "MS"})
    m = c.lump(["MS", "SM"], into="sulphide")
    assert m.names == ["RH", "sulphide"] and m.other == "sulphide" and m.colors == ["b", "0.6"]
    assert m.mapping == {"MS": "sulphide", "MSX": "sulphide", "SM": "sulphide"}
    assert m.decode(m.encode(["MSX", "QZ"])) == ["sulphide", "sulphide"]
    assert len(m) == 2 and "sulphide" in repr(m)


@pytest.mark.parametrize(
    "c",
    [
        cs.Categories(["a", "b"]),
        cs.Categories.from_values(list("aaabbbbc"), min_share=0.2, colors=["r", "g", "k"]),
    ],
)
def test_json_and_pickle_round_trip(c):
    for back in (cs.Categories.from_json(c.to_json()), pickle.loads(pickle.dumps(c))):
        assert back == c and back.to_json() == c.to_json()


def test_errors():
    c = cs.Categories(["a", "b"])
    with pytest.raises(cs.InvalidInput, match="6 labels are not categories: c, d, e, f, g$"):
        c.encode(list("abcdefgh"))
    for bad in (
        lambda: cs.Categories([]),
        lambda: cs.Categories(["a", "a"]),
        lambda: cs.Categories(["o", "a"], other="o"),
        lambda: cs.Categories(["a"], colors=["r", "g"]),
        lambda: cs.Categories(["a"], mapping={"x": "b"}),
        lambda: c.decode([2]),
        lambda: c.decode([0.5]),
        lambda: c.shares([0], weights=[-1.0]),
        lambda: c.lump(["z"]),
        lambda: cs.Categories.from_values([None]),
        lambda: cs.Categories.from_values(["a"], min_share=2.0),
        lambda: cs.Categories.from_json(cs.Search(1.0).to_json()),
    ):
        with pytest.raises(cs.InvalidInput):
            bad()


def test_plots_follow_the_scheme():
    c = cs.Categories(["b", "a"], other="other")
    codes = c.encode(["a", "b", "b", "z", None])
    _, ax = cs.plot.proportions(codes, scheme=c)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["b", "a", "other"]
    np.testing.assert_allclose([p.get_width() for p in ax.patches], [0.5, 0.25, 0.25])
    assert ax.patches[-1].get_facecolor()[:3] == pytest.approx((0.6, 0.6, 0.6))
    xy = np.c_[np.arange(5.0), np.zeros(5)]
    _, ax = cs.plot.category_swath(xy, codes, 10.0, axis="x", scheme=c)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["other", "a", "b"]


def rgba(color):
    return pytest.approx(matplotlib.colors.to_rgba(color))


def test_category_colors_map_code_i_to_color_i():
    colors = ["#112233", "#445566", "#778899"]
    cmap, norm = cs.plot.category_colors(cs.Categories(["a", "b", "c"], colors=colors))
    np.testing.assert_allclose(norm.boundaries, [-0.5, 0.5, 1.5, 2.5])
    for i, color in enumerate(colors):
        for code in (i - 0.49, i, i + 0.49):
            assert cmap(norm(code)) == rgba(color)
    cmap, _ = cs.plot.category_colors(cs.Categories(["a", "b"], other="rest"))
    assert cmap.N == 3 and cmap(2) == rgba("0.6")


def test_category_legend_names_patches_in_code_order():
    import matplotlib.pyplot as plt

    c = cs.Categories(["z", "a"], colors=["red", "blue", "0.3"], other="other")
    fig, ax = plt.subplots()
    for target in (ax, fig):
        legend = cs.plot.category_legend(c, target, ncol=3)
        assert [t.get_text() for t in legend.get_texts()] == ["z", "a", "other"]
        assert legend.legend_handles[1].get_facecolor() == rgba("blue")


def test_section_slab_and_boxplot_take_a_scheme():
    c = cs.Categories(["a", "b"], colors=["red", "blue"])
    bm = cs.BlockModel(origin=(0, 0, 0), size=(1, 1, 1), count=(2, 2, 1)).with_column("k", [0.0, 1, 1, 0])
    _, ax = cs.plot.section(bm, "k", scheme=c)
    assert ax.images[0].cmap(ax.images[0].norm(1)) == rgba("blue")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"] and len(ax.figure.axes) == 1
    points = np.c_[np.arange(4.0), np.zeros(4), np.zeros(4)]
    _, ax = cs.plot.slab(points, [1.0, 0, 1, 0], plane=((0, 0, 0), 90, 90), thickness=1, scheme=c)
    faces = ax.collections[-1].to_rgba(ax.collections[-1].get_array())
    assert faces[0] == rgba("blue") and faces[1] == rgba("red")
    assert ax.get_legend() is not None and len(ax.figure.axes) == 1
    _, ax = cs.plot.boxplot([1.0, 2, 3, 4], [1.0, 1, 0, np.nan], sort=True, scheme=c)
    assert [t.get_text().split("\n")[0] for t in ax.get_xticklabels()] == ["b", "a"]
    assert ax.patches[0].get_facecolor() == rgba((0, 0, 1, 0.6))
