"""Training: disaggregate the heat pump compressor's 15-minute mean power from the main meter.

Reads a year of the operator's inputs -- the main meter's factored energy register
and the EV socket's energy register -- and, as a training-only topic, the compressor
sub-meter's own power. The sub-meter is the label: it is read here and nowhere else,
because the deployed operator never receives it.

The fit runs in the driver: one year at 15 minutes is about 35 000 rows, which a
gradient-boosted tree ensemble fits in seconds.

Before the final fit the last VALIDATION of history is held out, the model fitted on
what precedes it and scored there, and that score logged as val_mae. That is a check
against data the session may read; the evaluation's own score over the test window is
recorded separately by Operator Lib.
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

# A bucket is a training row only if the label and the main meter both cover it.
MIN_LABEL_SAMPLES = 20
MIN_MAIN_SAMPLES = 5

# What the compressor can draw, for clipping an estimate.
P_MIN, P_MAX = -5.0, 4500.0

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


def _local_hour(buckets) -> np.ndarray:
    local = pd.to_datetime(np.asarray(buckets, dtype=float), unit="s", utc=True).tz_convert(F.LOCAL_TZ)
    return (local.hour + local.minute / 60.0).to_numpy(dtype=float)


def build_table(main_ts, main_vs, ev_ts, ev_vs, label_ts, label_vs) -> pd.DataFrame:
    labels = F.bucket_means(label_ts, label_vs)
    labels = labels[labels["count"] >= MIN_LABEL_SAMPLES]
    if labels.empty or len(main_ts) == 0:
        return pd.DataFrame()
    lo = F.bucket_of(np.nanmin(main_ts)) + F.CONTEXT_BUCKETS * F.BUCKET_S
    hi = F.bucket_of(np.nanmax(main_ts)) - (F.LOOKAHEAD_BUCKETS + 1) * F.BUCKET_S
    starts = labels.index.to_numpy(dtype=float)
    starts = starts[(starts >= lo) & (starts <= hi)]
    X = F.build_features(main_ts, main_vs, ev_ts, ev_vs, starts)
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


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    with logger.trace("read history"):
        inputs = [ds.to_pandas() for ds in provide_historic_data_local(TRAINING_WINDOW)]
        training = [ds.to_pandas() for ds in provide_historic_data_local(TRAINING_WINDOW, role="training")]

    main_ts, main_vs = _series(_frame_with(inputs, "main_energy"), "main_energy")
    ev_ts, ev_vs = _series(_frame_with(inputs, "ev_energy"), "ev_energy")
    label_ts, label_vs = _series(_frame_with(training, "target"), "target")

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "validation_days": VALIDATION.days,
        "rows_main": int(len(main_ts)),
        "rows_ev": int(len(ev_ts)),
        "rows_label": int(len(label_ts)),
        "context_buckets": F.CONTEXT_BUCKETS,
        "lookahead_buckets": F.LOOKAHEAD_BUCKETS,
        **{f"hgb_{k}": v for k, v in MODEL_PARAMS.items()},
    })

    if len(main_ts) == 0 or len(label_ts) == 0:
        # Explicitly nothing rather than a model fitted on no data.
        return None

    with logger.trace("features"):
        table = build_table(main_ts, main_vs, ev_ts, ev_vs, label_ts, label_vs)
    if table.empty:
        return None
    columns = [c for c in table.columns if c not in ("bucket", "target")]

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

    with logger.trace("fit"):
        model = CompressorPowerModel(_fit(table, columns), columns)
    logger.log_metrics({"fit_buckets": float(len(table)), "fit_mae_in_sample": _mae(model, table)})
    return model
