"""Offline unit tests for the toolkit (no network access needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai import econ, features, models, travel  # noqa: E402
from geoai.grid import Grid, block_reduce_mean, grid_around, grid_from_lonlat, lonlat_to_tm, tm_to_lonlat  # noqa: E402
from geoai.labels import wc_tiles  # noqa: E402
from geoai.stac import dry_season, collection_for  # noqa: E402


def test_projection_roundtrip():
    x, y = lonlat_to_tm(90.4, 23.78)
    lon, lat = tm_to_lonlat(x, y)
    assert abs(lon - 90.4) < 1e-9 and abs(lat - 23.78) < 1e-9


def test_grids_nest_across_resolutions():
    g20 = grid_from_lonlat(89, 22, 90, 23, 20)
    g1k = grid_from_lonlat(89, 22, 90, 23, 1000)
    assert g20.xmin == g1k.xmin and g20.ymax == g1k.ymax
    assert g20.width * 20 == g1k.width * 1000


def test_chunks_cover_grid_exactly():
    g = grid_around(90.4, 23.78, 5, 20)
    area = sum(sg.width * sg.height for _, _, sg in g.chunks(128))
    assert area == g.width * g.height


def test_block_reduce_mean_ignores_nan():
    a = np.array([[1, np.nan], [3, 4]], float)
    assert block_reduce_mean(a, 2)[0, 0] == pytest.approx(8 / 3)


def test_season_and_collection():
    assert dry_season(2023) == ("2022-11-15", "2023-03-31")
    assert collection_for(2019) == "sentinel-2-l2a" and collection_for(2020) == "sentinel-2-c1-l2a"


def test_worldcover_tile_names():
    assert wc_tiles((89.5, 22.5, 90.5, 23.5)) == ["N21E087", "N21E090"]


def test_feature_stack_shapes_and_indices():
    rng = np.random.default_rng(0)
    refl = rng.uniform(0.02, 0.4, (6, 32, 32)).astype(np.float32)
    F, names = features.feature_stack(refl)
    assert F.shape[0] == len(names) and F.shape[1:] == (32, 32)
    ndvi = F[names.index("ndvi")]
    assert np.allclose(ndvi, (refl[3] - refl[2]) / (refl[3] + refl[2]), atol=1e-6)


def test_block_folds_are_spatially_constant_within_blocks():
    f = models.block_folds(100, 100, res=20, block_m=400, k=5)
    assert (f[:20, :20] == f[0, 0]).all()
    # chunked computation agrees with full-grid computation
    sub = models.block_folds(40, 40, res=20, block_m=400, k=5, offset=(60, 20))
    assert (sub == f[60:100, 20:60]).all()


def test_travel_time_road_beats_offroad_and_water_blocks():
    g = Grid(0, 10_000, 100, 100, 100.0)  # 10 km square, 100 m cells
    water = np.zeros((100, 100))
    water[:, 50] = 1.0  # a river splitting the grid
    import geopandas as gpd
    from shapely.geometry import LineString

    roads = gpd.GeoDataFrame({"class": ["primary"]}, geometry=[LineString([(0, 5000), (10_000, 5000)])])
    fr = travel.friction_surface(g, roads, water)
    # destination at the west end of the road
    import geoai.travel as T

    orig = T.lonlat_to_tm
    T.lonlat_to_tm = lambda lon, lat: (lon, lat)
    try:
        tt = travel.travel_time_to(g, fr, 50.0, 5000.0, radius_cells=0)
    finally:
        T.lonlat_to_tm = orig
    on_road_east = tt[50, 99]
    assert on_road_east == pytest.approx(10 / 40, rel=0.05)  # 10 km at 40 km/h via the road bridge
    fr_nobridge = travel.friction_surface(g, roads.iloc[:0], water)
    T.lonlat_to_tm = lambda lon, lat: (lon, lat)
    try:
        tt2 = travel.travel_time_to(g, fr_nobridge, 50.0, 5000.0, radius_cells=0)
    finally:
        T.lonlat_to_tm = orig
    assert np.isinf(tt2[50, 99])  # no way across the river without a bridge


def test_feols_recovers_coefficient_with_two_way_fe():
    rng = np.random.default_rng(1)
    n_i, n_t = 300, 6
    i = np.repeat(np.arange(n_i), n_t)
    t = np.tile(np.arange(n_t), n_i)
    D = np.repeat(rng.uniform(0, 1, n_i), n_t)
    post = (t >= 3).astype(float)
    y = np.repeat(rng.normal(0, 3, n_i), n_t) + t * 0.7 + 2.0 * D * post + rng.normal(0, 0.5, n_i * n_t)
    df = pd.DataFrame({"y": y, "x": D * post, "i": i, "t": t, "cl": i // 10})
    res, _ = econ.feols(df, "y", ["x"], ["i", "t"], "cl")
    assert res.loc["x", "coef"] == pytest.approx(2.0, abs=0.15)
    assert res.loc["x", "se"] > 0


def test_mosaiks_feature_dimension():
    from geoai.mosaiks import RandomConvFeatures

    pytest.importorskip("torch")
    imgs = np.random.default_rng(0).uniform(0, 0.3, (20, 6, 16, 16)).astype(np.float32)
    rcf = RandomConvFeatures(n_filters=8, seed=0).fit(imgs)
    assert rcf.transform(imgs).shape == (20, 16)


def test_lei_classifies_expansion_types():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib

    ue = importlib.import_module("03_urban_expansion")
    b0 = np.zeros((40, 40), bool)
    b0[10:20, 10:20] = True
    b1 = b0.copy()
    b1[12:18, 20:23] = True  # attached to the old patch -> edge expansion
    b1[30:33, 30:33] = True  # isolated -> outlying
    s = ue.lei_shares(b0, b1)
    assert s["edge"] > 0 and s["outlying"] > 0 and s["infill"] == 0


def test_balanced_scene_selection_spreads_dates_and_prefers_c1():
    import datetime as dt
    from types import SimpleNamespace

    from geoai import composite

    def item(day, cloud, coll="sentinel-2-c1-l2a"):
        d = dt.datetime(2022, 11, 15) + dt.timedelta(days=day)
        return SimpleNamespace(datetime=d.replace(tzinfo=dt.timezone.utc), collection_id=coll,
                               properties={"eo:cloud_cover": cloud, "grid:code": "MGRS-46QBM"}, id="x_46QBM")

    # Many clear scenes early in the season, one cloudier scene late, and an L2A-only window.
    items = [item(d, 0.0) for d in range(0, 20, 5)] + [item(130, 30.0)] + [item(70, 5.0, "sentinel-2-l2a")]
    items.sort(key=lambda it: it.properties["eo:cloud_cover"])
    sel = composite.select_scenes(items, 6, year=2023)
    days = sorted((it.datetime.replace(tzinfo=None) - dt.datetime(2022, 11, 15)).days for it in sel)
    assert 130 in days and 70 in days  # late and gap-filled windows are represented
    old = composite.select_scenes(items, 3)  # least-cloudy rule takes only the early scenes
    assert all((it.datetime.replace(tzinfo=None) - dt.datetime(2022, 11, 15)).days < 20 for it in old)


def test_orbit_aware_selection_takes_scenes_from_both_orbits():
    import datetime as dt
    from types import SimpleNamespace

    from geoai import composite, stac

    def item(day, cloud, orbit):
        d = dt.datetime(2025, 11, 15, tzinfo=dt.timezone.utc) + dt.timedelta(days=day)
        uri = f"S2B_MSIL2A_{d:%Y%m%d}T043809_N0512_{orbit}_T45QYF_x.SAFE"
        return SimpleNamespace(datetime=d, collection_id=stac.C1, id="S2B_T45QYF_x",
                               properties={"eo:cloud_cover": cloud, "grid:code": "MGRS-45QYF", "s2:product_uri": uri})

    items = [item(d, 0.0, "R033") for d in range(0, 136, 5)] + [item(d + 2, 5.0, "R090") for d in range(0, 136, 5)]
    assert stac.rel_orbit(items[0]) == "R033"
    per_tile = {stac.rel_orbit(it) for it in composite.select_scenes(items, 6, year=2026)}
    per_orbit = [stac.rel_orbit(it) for it in composite.select_scenes(items, 6, year=2026, by_orbit=True)]
    assert per_tile == {"R033"}  # the cloud-free orbit wins every window
    assert per_orbit.count("R033") == 6 and per_orbit.count("R090") == 6


def test_qse_baseline_shares_clear_markets_and_null_shock_is_identity():
    from geoai import qse

    rng = np.random.default_rng(0)
    n = 30
    xy = rng.uniform(0, 100, (n, 2))
    t = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1)) / 40 + 0.2
    Y = rng.lognormal(0, 1, n)
    pi, _ = qse.trade_shares(Y, t ** -1.0)
    assert np.allclose(pi @ Y, Y, rtol=1e-8) and np.allclose(pi.sum(0), 1)
    lam = Y / Y.sum()
    r = qse.counterfactual(pi, Y, lam, np.ones((n, n)), sigma=5, alpha=0.75, eps=3)
    assert np.allclose(r["w_hat"], 1, atol=1e-7) and np.allclose(r["L_hat"], 1, atol=1e-7)
    assert r["welfare"] == pytest.approx(1.0, abs=1e-7)


def test_qse_lower_trade_costs_raise_welfare_and_attract_people():
    from geoai import qse

    rng = np.random.default_rng(1)
    n = 20
    xy = rng.uniform(0, 100, (n, 2))
    t = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1)) / 40 + 0.2
    Y = rng.lognormal(0, 0.5, n)
    lam = Y / Y.sum()
    pi, _ = qse.trade_shares(Y, t ** -1.0)
    t_new = t.copy()
    t_new[0, 1:] *= 0.5  # location 0 gets better connected
    t_new[1:, 0] *= 0.5
    phi_hat = (t_new / t) ** -1.0
    sr = qse.counterfactual(pi, Y, lam, phi_hat, sigma=5, alpha=0.75, eps=None)
    lr = qse.counterfactual(pi, Y, lam, phi_hat, sigma=5, alpha=0.75, eps=3)
    assert sr["converged"] and lr["converged"]
    assert sr["welfare"] > 1 and lr["welfare"] > 1
    assert lr["L_hat"][0] > 1 and lr["L_hat"][0] == lr["L_hat"].max()
    assert (lam * lr["L_hat"]).sum() == pytest.approx(1.0)


def test_qse_nontraded_sector_scales_short_run_gains_by_traded_share():
    from geoai import qse

    rng = np.random.default_rng(2)
    n = 25
    xy = rng.uniform(0, 100, (n, 2))
    t = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1)) / 40 + 0.2
    Y = rng.lognormal(0, 0.5, n)
    lam = Y / Y.sum()
    pi, _ = qse.trade_shares(Y, t ** -1.0)
    t_new = t.copy()
    t_new[:5, 5:] *= 0.6
    t_new[5:, :5] *= 0.6
    phi_hat = (t_new / t) ** -1.0
    one = qse.counterfactual(pi, Y, lam, phi_hat, sigma=5, alpha=0.75)
    two = qse.counterfactual(pi, Y, lam, phi_hat, sigma=5, alpha=0.75, alpha_N=0.30)
    # In the short run wages and traded prices do not depend on alpha_N, so log welfare scales by alpha_T / alpha.
    assert np.allclose(np.log(two["V_hat"]), (0.45 / 0.75) * np.log(one["V_hat"]), atol=1e-7)
    lr1 = qse.counterfactual(pi, Y, lam, phi_hat, sigma=5, alpha=0.75, eps=3)
    lr2 = qse.counterfactual(pi, Y, lam, phi_hat, sigma=5, alpha=0.75, eps=3, alpha_N=0.30)
    assert 1 < lr2["welfare"] < lr1["welfare"]
    assert np.abs(np.log(lr2["L_hat"])).max() < np.abs(np.log(lr1["L_hat"])).max()


def test_power_and_minimum_detectable_effect_agree():
    se = 0.05
    mde = econ.minimum_detectable_effect(se, target_power=0.80)
    assert abs(mde / se - 2.8016) < 1e-3
    assert abs(econ.power(mde, se) - 0.80) < 1e-3  # the neglected tail is below 1e-4
    assert abs(econ.power(0.0, se) - 0.05) < 1e-9
    assert econ.power(0.5 * mde, se) < econ.power(mde, se) < econ.power(2 * mde, se)


def test_weighted_groups_hold_equal_weight_and_are_ordered():
    rng = np.random.default_rng(0)
    v, w = rng.normal(size=2000), rng.uniform(0.5, 1.5, size=2000)
    g = econ.weighted_groups(v, w, k=5)
    shares = np.array([w[g == i].sum() for i in range(1, 6)]) / w.sum()
    assert set(g) == {1, 2, 3, 4, 5} and np.allclose(shares, 0.2, atol=0.002)
    assert all(v[g == i].max() <= v[g == i + 1].min() for i in range(1, 5))

