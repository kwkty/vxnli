from typing import Callable, Optional, Tuple

import altair as alt
import pandas as pd

from vxnli._vega_zero import VegaZero
from vxnli.errors import InputError


class Plot:
    def __init__(
        self,
        model: Optional[Callable[..., str]] = None,
    ) -> None:
        if model is None:
            from vxnli.models.v0.model import Model

            model = Model()

        self.model = model

    def __call__(self, *args, **kwargs) -> alt.Chart:
        data, args, kwargs = self._parse_args_and_kwargs(args, kwargs)
        vega_zero = self.model(data, *args, **kwargs)

        vega_zero = VegaZero.from_str(vega_zero)

        vega_lite = vega_zero.to_vega_lite(data)
        vega_lite = alt.Chart.from_dict(vega_lite)

        return vega_lite

    def _parse_args_and_kwargs(
        self, args: Tuple, kwargs: dict
    ) -> Tuple[pd.DataFrame, Tuple, dict]:
        d1, args = self._parse_args(args)
        d2, kwargs = self._parse_kwargs(kwargs)

        if d1 is not None and d2 is not None:
            raise InputError("Don't give multiple pandas dataframes")

        if d1 is None and d2 is None:
            raise InputError("Provide pandas dataframe somewhere in arguments")

        return d1 if d2 is None else d2, args, kwargs

    def _parse_args(self, args: Tuple) -> Tuple[Optional[pd.DataFrame], Tuple]:
        data = [(i, arg) for i, arg in enumerate(args) if isinstance(arg, pd.DataFrame)]

        if len(data) > 1:
            raise InputError("Don't give multiple pandas dataframes")

        if len(data) == 0:
            return None, args

        i, data = data[0]

        return data, args[:i] + args[i:]

    def _parse_kwargs(self, kwargs: dict) -> Tuple[Optional[pd.DataFrame], dict]:
        data = [(k, v) for k, v in kwargs.items() if isinstance(v, pd.DataFrame)]

        if len(data) > 1:
            raise InputError("Don't give multiple pandas dataframes")

        if len(data) == 0:
            return None, kwargs

        key, data = data[0]

        return data, {k: v for k, v in data.items() if k != key}
