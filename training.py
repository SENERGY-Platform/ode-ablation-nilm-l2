"""Training: disaggregate the heat pump compressor's 15-minute mean power from the main meter.

Reads a year of the operator's inputs -- the main meter's factored energy register,
the EV socket's energy register where it is an input, and outdoor temperature from
the open-meteo history import -- and, as a training-only topic, the compressor
sub-meter's own power. The sub-meter is the label: it is read here and nowhere else,
because the deployed operator never receives it.

Only heating-season buckets (local months October to April) become training rows.
The compressor's behaviour outside the heating season -- rare runs, hot water only --
and the site's summer base load are unlike the months the operator is evaluated on,
and a model fitted to the whole year learns mostly from them.

The fit runs in the driver: a heating season at 15 minutes is about 20 000 rows, which
a gradient-boosted tree ensemble fits in seconds.

Before the final fit the last VALIDATION of history is held out, the model fitted on
what precedes it and scored there, and that score logged as val_mae. That is a check
against data the session may read; the evaluation's own score over the test window is
recorded separately by Operator Lib.

The hold-out also carries a diagnostic that is independent of the model: how strongly
the bucket-to-bucket change in main-meter power follows the change in compressor
power, at lags of -4..+4 buckets. A compressor start or stop is a 3.3 kW step; if the
main meter sees it, the correlation peaks at lag 0. A peak elsewhere is a time offset
between the two meters, and no peak at all means the main meter does not carry the
compressor visibly at this resolution.
"""

import datetime
import typing

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel
from sklearn.ensemble import HistGradientBoostingRegressor

from operator_lib.util.helpers import provide_historic_data_local, TrainMlflowLogger

import features as F


# How much history one training pass reads. A year covers one full heating season
# before the test window, including last February.
TRAINING_WINDOW = datetime.timedelta(days=365)
VALIDATION = datetime.timedelta(days=31)

# Local months whose buckets become training rows.
HEATING_MONTHS = (10, 11, 12, 1, 2, 3, 4)

# A bucket is a training row only if the label and the main meter both cover it.
MIN_LABEL_SAMPLES = 20
MIN_MAIN_SAMPLES = 5

# What the compressor can draw, for clipping an estimate.
P_MIN, P_MAX = -5.0, 4500.0

# Lags, in buckets, of the step-correlation diagnostic.
DIAGNOSTIC_LAGS = range(-4, 5)

# The v1 settings. A smaller model (15 leaves, 300 per leaf, 300 iterations) was worse
# on the training year, on January and on February alike, so the gap between the
# in-sample and the hold-out score was not capacity.
MODEL_PARAMS = dict(
    loss="absolute_error",
    learning_rate=0.05,
    max_iter=600,
    max_leaf_nodes=63,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=0,
)


class CompressorPowerModel(PythonModel):
    """The model MLflow registers and op.py loads.

    predict() takes a frame of features.build_features rows and returns the estimated
    15-minute mean compressor power in watts, one per row.
    """

    def __init__(self, regressor, columns: typing.List[str]) -> None:
        self.regressor = regressor
        self.columns = list(columns)

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        X = pd.DataFrame(payload).reindex(columns=self.columns).to_numpy(dtype=float)
        return np.clip(self.regressor.predict(X), P_MIN, P_MAX)


def _frame_with(frames, column):
    for frame in frames:
        if column in frame.columns:
            return frame
    return None


def _series(frame, column):
    if frame is None or frame.empty:
        return np.array([]), np.array([])
    return F.to_seconds(frame["time"]), F.to_values(frame[column])


def _local(buckets) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(buckets, dtype=float), unit="s", utc=True).tz_convert(F.LOCAL_TZ)


def _local_hour(buckets) -> np.ndarray:
    local = _local(buckets)
    return (local.hour + local.minute / 60.0).to_numpy(dtype=float)


def build_table(main_ts, main_vs, ev_ts, ev_vs, label_ts, label_vs, temp_ts=None, temp_vs=None) -> pd.DataFrame:
    labels = F.bucket_means(label_ts, label_vs)
    labels = labels[labels["count"] >= MIN_LABEL_SAMPLES]
    if labels.empty or len(main_ts) == 0:
        return pd.DataFrame()
    lo = F.bucket_of(np.nanmin(main_ts)) + F.CONTEXT_BUCKETS * F.BUCKET_S
    hi = F.bucket_of(np.nanmax(main_ts)) - (F.LOOKAHEAD_BUCKETS + 1) * F.BUCKET_S
    starts = labels.index.to_numpy(dtype=float)
    starts = starts[(starts >= lo) & (starts <= hi)]
    starts = starts[np.isin(_local(starts).month, HEATING_MONTHS)]
    X = F.build_features(main_ts, main_vs, ev_ts, ev_vs, starts, temp_ts, temp_vs)
    X["bucket"] = starts
    X["target"] = labels.loc[starts, "mean"].to_numpy()
    X = X[(X["n_main"] >= MIN_MAIN_SAMPLES) & np.isfinite(X["P"])]
    return X.reset_index(drop=True)


def _fit(X: pd.DataFrame, columns):
    reg = HistGradientBoostingRegressor(**MODEL_PARAMS)
    reg.fit(X[columns].to_numpy(dtype=float), X["target"].to_numpy(dtype=float))
    return reg


def _mae(model: CompressorPowerModel, X: pd.DataFrame) -> float:
    if X.empty:
        return float("nan")
    return float(np.mean(np.abs(model.predict(None, X) - X["target"].to_numpy())))


def _lag_correlations(part: pd.DataFrame, prefix: str) -> typing.Dict[str, float]:
    """Pearson correlation of d_prev (main power, this bucket minus the previous one)
    with the same difference of the target, shifted by k buckets.

    The target difference is taken only between buckets that are actually adjacent,
    so a gap in the label never produces a spurious step. Lag k pairs the main-meter
    step in bucket t with the compressor step in bucket t + k.
    """
    out: typing.Dict[str, float] = {}
    if part.empty:
        return out
    target = pd.Series(part["target"].to_numpy(), index=part["bucket"].to_numpy())
    d_target = target - target.reindex(target.index - F.BUCKET_S).to_numpy()
    d_main = pd.Series(part["d_prev"].to_numpy(), index=part["bucket"].to_numpy())
    for k in DIAGNOSTIC_LAGS:
        shifted = d_target.reindex(d_main.index + k * F.BUCKET_S).to_numpy()
        a, b = d_main.to_numpy(), shifted
        ok = np.isfinite(a) & np.isfinite(b)
        name = f"{prefix}_lag{k:+d}".replace("+", "p").replace("-", "m")
        if ok.sum() < 30 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            out[name] = float("nan")
        else:
            out[name] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return out


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    with logger.trace("read history"):
        inputs = [ds.to_pandas() for ds in provide_historic_data_local(TRAINING_WINDOW)]
        training = [ds.to_pandas() for ds in provide_historic_data_local(TRAINING_WINDOW, role="training")]

    main_ts, main_vs = _series(_frame_with(inputs, "main_energy"), "main_energy")
    ev_ts, ev_vs = _series(_frame_with(inputs, "ev_energy"), "ev_energy")
    temp_ts, temp_vs = _series(_frame_with(inputs, "temperature"), "temperature")
    label_ts, label_vs = _series(_frame_with(training, "target"), "target")

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "validation_days": VALIDATION.days,
        "heating_months": ",".join(str(m) for m in HEATING_MONTHS),
        "context_buckets": F.CONTEXT_BUCKETS,
        "lookahead_buckets": F.LOOKAHEAD_BUCKETS,
        **{f"hgb_{k}": v for k, v in MODEL_PARAMS.items()},
    })
    # Also as metrics: the run summary carries metrics, and these say what training saw.
    logger.log_metrics({
        "rows_main": float(len(main_ts)),
        "rows_ev": float(len(ev_ts)),
        "rows_temp": float(len(temp_ts)),
        "rows_label": float(len(label_ts)),
    })

    if len(main_ts) == 0 or len(label_ts) == 0:
        # Explicitly nothing rather than a model fitted on no data.
        return None

    with logger.trace("features"):
        table = build_table(main_ts, main_vs, ev_ts, ev_vs, label_ts, label_vs, temp_ts, temp_vs)
    if table.empty:
        return None
    columns = [c for c in table.columns if c not in ("bucket", "target")]
    logger.log_metrics({"temp_coverage": float(np.isfinite(table["temp_now"]).mean())})

    split = table["bucket"].max() + F.BUCKET_S - VALIDATION.total_seconds()
    train_part = table[table["bucket"] < split].reset_index(drop=True)
    val_part = table[table["bucket"] >= split].reset_index(drop=True)

    with logger.trace("validate"):
        held = CompressorPowerModel(_fit(train_part, columns), columns)
        hour = _local_hour(val_part["bucket"])
        night = (hour >= 22) | (hour < 6)
        logger.log_metrics({
            "train_buckets": float(len(train_part)),
            "val_buckets": float(len(val_part)),
            "val_mae": _mae(held, val_part),
            "val_mae_night": _mae(held, val_part[night]),
            "val_mae_day": _mae(held, val_part[~night]),
            "val_mae_predict_zero": float(np.mean(np.abs(val_part["target"]))) if len(val_part) else float("nan"),
        })

    with logger.trace("diagnose"):
        logger.log_metrics({
            **_lag_correlations(val_part, "val_corr_dP_dtarget"),
            **_lag_correlations(val_part[night].reset_index(drop=True), "val_corr_night"),
        })

    with logger.trace("fit"):
        model = CompressorPowerModel(_fit(table, columns), columns)
    logger.log_metrics({"fit_buckets": float(len(table)), "fit_mae_in_sample": _mae(model, table)})
    return model
