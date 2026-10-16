"""The operator: estimate the heat pump compressor's 15-minute mean power from the main meter.

Inputs, by selector:
  main  -- main_energy: the main meter's register obis_1_8_0_f, kWh on the primary
           side (transformer factor 50 applied by the meter's own firmware).
  ev    -- ev_energy: the EV charging socket's energy register, kWh. A covariate: the
           socket draws about as much as the compressor and would otherwise be
           mistaken for it.

Output: {"prediction": W}, one per 15-minute bucket, stamped with the bucket's start.
An estimate is published once a main-meter sample has arrived LOOKAHEAD_BUCKETS
after the bucket's end, because the features look that far ahead. Every bucket gets
an estimate, including ones the model is unsure of.

The compressor's own sub-meter is a training label only; it never reaches infer().
"""

import bisect
import datetime
import typing

import numpy as np
from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

import features as F
from training import train_model


class CustomConfig(Config):
    # Retrain at most this often, in seconds. A training pass reads a year of three
    # series, so weekly rather than daily.
    retrain_after_s = 7 * 86400


# How much raw history the buffers keep behind the next bucket to be estimated.
_KEEP_S = (F.CONTEXT_BUCKETS + 1) * F.BUCKET_S + 600


def _prune(ts: typing.List[float], vs: typing.List[float], cutoff: float):
    """Drop samples before cutoff, keeping what the register knots still need.

    A knot sits where the register took a new value, at the midpoint with the sample
    before it. So the run of equal values reaching up to the cutoff keeps its first
    sample and that sample's predecessor, and the samples in between -- which are not
    knots -- are dropped. The knots left are the ones training computed over the
    full history.
    """
    i = bisect.bisect_left(ts, cutoff)
    if i < 3:
        return ts, vs
    j = i - 1
    while j > 0 and vs[j - 1] == vs[i - 1]:
        j -= 1
    keep = sorted({max(j - 1, 0), j, i - 1})
    return [ts[k] for k in keep] + ts[i:], [vs[k] for k in keep] + vs[i:]


def _append(ts: typing.List[float], vs: typing.List[float], t: float, v: float):
    if not ts or t > ts[-1]:
        ts.append(t)
        vs.append(v)
    elif t == ts[-1]:
        vs[-1] = v
    else:
        k = bisect.bisect_left(ts, t)
        if k < len(ts) and ts[k] == t:
            vs[k] = v
        else:
            ts.insert(k, t)
            vs.insert(k, v)


class Operator(MLOperator):
    configType = CustomConfig

    selectors = [
        Selector({"name": "main", "args": ["main_energy"]}),
        Selector({"name": "ev", "args": ["ev_energy"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns, so anything set after it is missing
        # during the replay and overwrites what train() just recorded.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self.main_t: typing.List[float] = []
        self.main_v: typing.List[float] = []
        self.ev_t: typing.List[float] = []
        self.ev_v: typing.List[float] = []
        self.next_bucket: typing.Optional[float] = None
        super().init(*args, **kwargs)

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        t = timestamp.timestamp()

        if selector == "ev":
            value = data.get("ev_energy")
            if value is not None:
                _append(self.ev_t, self.ev_v, t, float(value))
            return None, None, None

        value = data.get("main_energy")
        if value is None:
            return None, None, None
        _append(self.main_t, self.main_v, t, float(value))

        if self.next_bucket is None:
            # The bucket the first sample lands in has no start edge; begin after it.
            self.next_bucket = float(F.bucket_of(t)) + F.BUCKET_S

        ready_at = self.next_bucket + (1 + F.LOOKAHEAD_BUCKETS) * F.BUCKET_S + F.READY_MARGIN_S
        if model is None or self.main_t[-1] < ready_at:
            return None, None, None

        start = self.next_bucket
        self.next_bucket += F.BUCKET_S

        X = F.build_features(
            np.asarray(self.main_t), np.asarray(self.main_v),
            np.asarray(self.ev_t), np.asarray(self.ev_v),
            np.array([start]),
        )
        prediction = float(np.asarray(model.predict(X)).ravel()[0])

        cutoff = self.next_bucket - _KEEP_S
        self.main_t, self.main_v = _prune(self.main_t, self.main_v, cutoff)
        self.ev_t, self.ev_v = _prune(self.ev_t, self.ev_v, cutoff)

        bucket_time = datetime.datetime.fromtimestamp(start, tz=datetime.timezone.utc)
        return bucket_time, {"prediction": prediction}, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
