"""Per-bucket features from the main meter's energy register, shared by training and inference.

The main meter publishes a cumulative register, obis_1_8_0_f: kWh on the primary side,
with the transformer factor of 50 already applied (it is exactly 50 x obis_1_8_0, and
the meter's own `factor` field reads 50). Its instantaneous power channel, obis_16_7,
carries no values at all, so power has to come from the register.

The register moves in steps of 0.25 kWh. Differencing it at the two edges of a
15-minute bucket therefore quantises the bucket's mean power to 1 kW -- five times
the error budget. What the register does carry more finely is *when* it stepped: a
step lands between two consecutive samples, so placing it at their midpoint and
interpolating linearly between steps recovers the energy at any instant to well
under one step. Every power below is read off that curve.

Outdoor temperature (open-meteo history, hourly, stamped with the hour it describes)
is the one covariate that changes from one winter day to the next and drives the
compressor's duty cycle. Two features come from it, both as of the bucket's end: the
latest hourly value, and the mean over the trailing 24 hours.

build_features computes the feature rows for any set of bucket starts from raw
samples. Training calls it over a year of history, infer() over buffered messages for
one bucket at a time: the same code on both paths, so the model never sees a feature
at inference that was computed differently in training.

Buckets are fixed 15-minute intervals truncated from the Unix epoch in UTC, which is
how Operator Lib's scorer buckets both the target and a prediction's result time.
"""

import warnings

import numpy as np
import pandas as pd

BUCKET_S = 900
SUB_S = 300

# Context the features look at: 2 h before the bucket, 30 min after it. The look-ahead
# is why an estimate is published half an hour after its bucket closes.
CONTEXT_BUCKETS = 8
LOOKAHEAD_BUCKETS = 2

# A bucket is computed once a main-meter sample this far past the end of its
# look-ahead has arrived, so that a register step beyond the last grid point exists.
READY_MARGIN_S = 300

LOCAL_TZ = "Europe/Berlin"

# Plausibility bounds for repairing a counter: a rise faster than max_kw, or a drop
# larger than drop_tol, is a discontinuity (a reset, a glitch, a replaced meter) and
# carries no energy. Smaller drops are jitter and are absorbed by a running maximum.
MAIN_MAX_KW = 400.0
MAIN_DROP_TOL = 0.5
EV_MAX_KW = 30.0
EV_DROP_TOL = 0.05

# Temperature: the trailing window of the mean, and how old the latest value may be
# before it counts as missing rather than current.
TEMP_MEAN_S = 24 * 3600
TEMP_MAX_AGE_S = 3 * 3600
# How much temperature history infer() keeps.
TEMP_KEEP_S = TEMP_MEAN_S + 2 * 3600

_FIRST = -CONTEXT_BUCKETS * BUCKET_S
_LAST = (1 + LOOKAHEAD_BUCKETS) * BUCKET_S
GRID = np.arange(_FIRST, _LAST + SUB_S, SUB_S, dtype=float)          # offsets from bucket start
SUB_SLOTS = np.arange(_FIRST // SUB_S, _LAST // SUB_S)                 # 5-min slots, 0..2 = the bucket
B15 = np.arange(-CONTEXT_BUCKETS, LOOKAHEAD_BUCKETS + 1)               # 15-min slots, 0 = the bucket
_I0 = -_FIRST // SUB_S                                                 # grid index of the bucket start
_STEP = BUCKET_S // SUB_S
_B0 = CONTEXT_BUCKETS                                                  # p15 column of the bucket
EV_B15 = np.arange(-2, 3)

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def to_seconds(times) -> np.ndarray:
    """Unix seconds from anything pandas reads as a time; naive values are UTC."""
    t = pd.to_datetime(pd.Series(times).reset_index(drop=True), utc=True, errors="coerce")
    return (t - _EPOCH).dt.total_seconds().to_numpy(dtype=float)


def to_values(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce").to_numpy(dtype=float)


def bucket_of(ts: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(ts, dtype=float) / BUCKET_S) * BUCKET_S


def _repair(ts: np.ndarray, vs: np.ndarray, max_kw: float, drop_tol: float) -> np.ndarray:
    if len(vs) < 2:
        return vs.copy()
    d = np.diff(vs)
    dt = np.maximum(np.diff(ts), 1.0)
    discontinuity = (d < -drop_tol) | (d > max_kw * dt / 3600.0 + drop_tol)
    removed = np.concatenate([[0.0], np.cumsum(np.where(discontinuity, d, 0.0))])
    return np.maximum.accumulate(vs - removed)


def counter_knots(ts, vs, max_kw: float, drop_tol: float):
    """The knots of a piecewise-linear energy curve through a quantised register.

    Each knot sits at the midpoint between the sample where the register took a new
    value and the sample before it, carrying that new value.
    """
    ts = np.asarray(ts, dtype=float)
    vs = np.asarray(vs, dtype=float)
    ok = np.isfinite(ts) & np.isfinite(vs)
    ts, vs = ts[ok], vs[ok]
    if len(ts) == 0:
        return ts, vs
    order = np.argsort(ts, kind="stable")
    ts, vs = ts[order], vs[order]
    last_of_time = np.r_[ts[1:] != ts[:-1], True]
    ts, vs = ts[last_of_time], vs[last_of_time]
    vs = _repair(ts, vs, max_kw, drop_tol)
    change = np.r_[True, vs[1:] != vs[:-1]]
    kt = ts.copy()
    kt[1:] = np.where(change[1:], 0.5 * (ts[1:] + ts[:-1]), ts[1:])
    return kt[change], vs[change]


def _energy_at(kt, kv, points, clamp: bool) -> np.ndarray:
    if len(kt) == 0:
        return np.full(points.shape, np.nan)
    e = np.interp(points.ravel(), kt, kv).reshape(points.shape)
    if not clamp:
        e[(points < kt[0]) | (points > kt[-1])] = np.nan
    return e


def _nan(fn, a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def _temperature(temp_ts, temp_vs, at: np.ndarray):
    """Latest value at or before `at` (missing if older than TEMP_MAX_AGE_S), and the
    mean of the values in (at - TEMP_MEAN_S, at]."""
    ts = np.asarray(temp_ts if temp_ts is not None else [], dtype=float)
    vs = np.asarray(temp_vs if temp_vs is not None else [], dtype=float)
    ok = np.isfinite(ts) & np.isfinite(vs)
    ts, vs = ts[ok], vs[ok]
    now = np.full(at.shape, np.nan)
    mean = np.full(at.shape, np.nan)
    if len(ts) == 0:
        return now, mean
    order = np.argsort(ts, kind="stable")
    ts, vs = ts[order], vs[order]
    hi = np.searchsorted(ts, at, side="right")
    lo = np.searchsorted(ts, at - TEMP_MEAN_S, side="right")
    last = hi - 1
    has = last >= 0
    fresh = has & ((at - ts[np.maximum(last, 0)]) <= TEMP_MAX_AGE_S)
    now[fresh] = vs[last[fresh]]
    csum = np.concatenate([[0.0], np.cumsum(vs)])
    count = hi - lo
    filled = count > 0
    mean[filled] = (csum[hi[filled]] - csum[lo[filled]]) / count[filled]
    return now, mean


def build_features(main_ts, main_vs, ev_ts, ev_vs, starts, temp_ts=None, temp_vs=None) -> pd.DataFrame:
    """One feature row per bucket start (Unix seconds, multiples of BUCKET_S).

    main_*: the main meter's factored register, kWh. ev_*: the EV socket's energy
    register, kWh, or empty arrays when there is none. temp_*: outdoor temperature,
    degrees C, stamped with the hour it describes, or None.
    """
    starts = np.asarray(starts, dtype=float)
    pts = starts[:, None] + GRID[None, :]

    mk, mv = counter_knots(main_ts, main_vs, MAIN_MAX_KW, MAIN_DROP_TOL)
    em = _energy_at(mk, mv, pts, clamp=False)
    p5 = np.diff(em, axis=1) * (3600.0 / SUB_S) * 1000.0
    p15 = np.diff(em[:, ::_STEP], axis=1) * (3600.0 / BUCKET_S) * 1000.0
    P = p15[:, _B0]

    ek, ev = counter_knots(ev_ts, ev_vs, EV_MAX_KW, EV_DROP_TOL)
    ee = _energy_at(ek, ev, pts, clamp=True)
    ev5 = np.diff(ee, axis=1) * (3600.0 / SUB_S) * 1000.0
    ev15 = np.diff(ee[:, ::_STEP], axis=1) * (3600.0 / BUCKET_S) * 1000.0
    r15 = p15 - ev15
    R = r15[:, _B0]

    f = {}
    for i, k in enumerate(SUB_SLOTS):
        f[f"p5_{k:+d}"] = p5[:, i]
    for i, j in enumerate(B15):
        f[f"p15_{j:+d}"] = p15[:, i]
    f["P"] = P
    f["P_minus_min_past"] = P - _nan(np.nanmin, p15[:, :_B0], 1)
    f["P_minus_median_ctx"] = P - _nan(np.nanmedian, p15, 1)
    f["d_prev"] = P - p15[:, _B0 - 1]
    f["d_next"] = p15[:, _B0 + 1] - P
    around = p5[:, _I0 - 3:_I0 + 6]
    steps = np.diff(around, axis=1)
    f["step_max"] = _nan(np.nanmax, steps, 1)
    f["step_min"] = _nan(np.nanmin, steps, 1)
    f["sub_range"] = _nan(np.nanmax, p5[:, _I0:_I0 + 3], 1) - _nan(np.nanmin, p5[:, _I0:_I0 + 3], 1)

    for j in EV_B15:
        f[f"ev15_{j:+d}"] = ev15[:, _B0 + j]
    for k in range(-1, 4):
        f[f"ev5_{k:+d}"] = ev5[:, _I0 + k]
    for i, j in enumerate(B15):
        f[f"r15_{j:+d}"] = r15[:, i]
    f["R"] = R
    f["R_minus_min_past"] = R - _nan(np.nanmin, r15[:, :_B0], 1)
    f["R_minus_median_ctx"] = R - _nan(np.nanmedian, r15, 1)

    t_now, t_mean = _temperature(temp_ts, temp_vs, starts + BUCKET_S)
    f["temp_now"] = t_now
    f["temp_24h"] = t_mean

    local = pd.to_datetime(starts, unit="s", utc=True).tz_convert(LOCAL_TZ)
    f["hour"] = (local.hour + local.minute / 60.0).to_numpy(dtype=float)
    f["dow"] = local.dayofweek.to_numpy(dtype=float)
    f["weekend"] = (local.dayofweek >= 5).astype(float)

    main_sorted = np.sort(np.asarray(main_ts, dtype=float)[np.isfinite(np.asarray(main_ts, dtype=float))])
    f["n_main"] = (np.searchsorted(main_sorted, starts + BUCKET_S) - np.searchsorted(main_sorted, starts)).astype(float)

    return pd.DataFrame(f)


def bucket_means(ts, vals) -> pd.DataFrame:
    """Per-bucket mean and sample count, the way the scorer computes the target."""
    ts = np.asarray(ts, dtype=float)
    vals = np.asarray(vals, dtype=float)
    ok = np.isfinite(ts) & np.isfinite(vals)
    df = pd.DataFrame({"bucket": bucket_of(ts[ok]), "v": vals[ok]})
    return df.groupby("bucket")["v"].agg(["mean", "count"])
