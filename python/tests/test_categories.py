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
        a = cs.Categories.from_values(raw, w, min_share)
        order = rng.permutation(raw.size)
        b = cs.Categories.from_values(raw[order], w[order], min_share)
        assert a == b and a.to_json() == b.to_json()
        if a.other is not None:
            assert a.names[-1] == a.other


def test_integers_sort_numerically_and_integral_floats_are_integers():
    c = cs.Categories.from_values([10.0, 9, "100", np.nan, None, 9.0])
    assert c.names == ["9", "10", "100"] and c.other is None
    np.testing.assert_array_equal(c.encode(pa.array([9, None, 100])), [0, np.nan, 2])


def test_shares_sum_to_one_and_other_holds_the_lumped():
    for raw, w in draws():
        full = cs.Categories.from_values(raw, w)
        full_shares = full.shares(full.encode(raw), w)
        min_share = rng.uniform(0, 0.3)
        c = cs.Categories.from_values(raw, w, min_share)
        shares = c.shares(c.encode(raw), w)
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
        lambda: c.shares([0], [-1.0]),
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
