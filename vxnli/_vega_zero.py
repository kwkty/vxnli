"""A VegaZero parser

The ncNet authors provide a vega zero parser.

https://github.com/Thanksyy/ncNet/blob/19e852368228ad251a28623950524a364ee95260/utilities/vis_rendering.py

However, unfortunately it has several bugs.
Firstly, I planned to fix them on the original code, but honestly speaking, the code quality is not enough to do that.
So I decided to refactor (re-implement) the parser.

In the end, I could solve some bugs, but there are still some problems.
Check ../notebooks/01_eda.ipynb for the details.
"""

import dataclasses
import re

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import pandas as pd

from vxnli.errors import VegaZeroError


@dataclasses.dataclass
class VegaZeroEncoding:
    x: str
    y: str
    y_aggregate: Optional[str] = None
    color: Optional[str] = None

    _PATTERN_01: ClassVar[re.Pattern] = re.compile(r"x (\S+) y aggregate (\S+) (.+)")
    _PATTERN_02: ClassVar[re.Pattern] = re.compile(r"(\S+) color (\S+)")

    @classmethod
    def parse(cls, vega_zero_encoding_str: str) -> "VegaZeroEncoding":
        match = cls._PATTERN_01.match(vega_zero_encoding_str)

        if match is None:
            raise VegaZeroError(
                f"Failed to parse vega_zero_encoding_str: {vega_zero_encoding_str}"
            )

        x, y_aggregate, y = match.groups()

        if y_aggregate == "none":
            y_aggregate = None

        match = cls._PATTERN_02.match(y)

        if match is None:
            color = None
        else:
            y, color = match.groups()

        return cls(
            x=x,
            y=y,
            y_aggregate=y_aggregate,
            color=color,
        )

    def __str__(self):
        y_aggregate = "none" if self.y_aggregate is None else self.y_aggregate
        color = "" if self.color is None else f" color {self.color}"

        return f"x {self.x} y aggregate {y_aggregate} {self.y}" + color


@dataclasses.dataclass
class VegaZeroTransform:
    filter: Optional[str] = None
    group: Optional[str] = None
    bin: Optional[Tuple[str, str]] = None
    sort: Optional[Tuple[str, str]] = None
    topk: Optional[int] = None

    @classmethod
    def _cast_types(cls, **kwargs: Dict[str, str]) -> Dict[str, Any]:
        if "bin" in kwargs:
            bin_ = kwargs["bin"].split()

            if len(bin_) != 3 or bin_[1] != "by":
                raise VegaZeroError(f"Invalid transform.bin format: {kwargs['bin']}")

            kwargs["bin"] = (bin_[0], bin_[2])

        if "sort" in kwargs:
            kwargs["sort"] = tuple(kwargs["sort"].rsplit(maxsplit=1))

            if len(kwargs["sort"]) == 1:
                kwargs["sort"] = (kwargs["sort"][0], "asc")

        if "topk" in kwargs:
            kwargs["topk"] = int(kwargs["topk"])

        return kwargs

    @classmethod
    def parse(cls, vega_zero_transform_str: str) -> "VegaZeroTransform":
        keywords = dataclasses.fields(cls)
        keywords = frozenset(k.name for k in keywords)

        tokens = vega_zero_transform_str.split()
        tokens = (token for token in tokens if token != "")

        spec = {}

        keyword, stack = next(tokens), []

        if keyword not in keywords:
            raise VegaZeroError("Invalid syntax")

        for token in tokens:
            if token in keywords:
                spec[keyword] = " ".join(stack)

                keyword, stack = token, []
            else:
                stack.append(token)

        spec[keyword] = " ".join(stack)
        spec = cls._cast_types(**spec)

        return cls(**spec)

    def __str__(self) -> str:
        transforms = []

        if self.filter is not None:
            transforms.append(f"filter {self.filter}")

        if self.group is not None:
            transforms.append(f"group {self.group}")

        if self.sort is not None:
            transforms.append(f"sort {self.sort[0]} {self.sort[1]}")

        if self.bin is not None:
            transforms.append(f"bin {self.bin[0]} by {self.bin[1]}")

        if self.topk is not None:
            transforms.append(f"topk {self.topk}")

        return " ".join(transforms)


@dataclasses.dataclass
class VegaZero:
    mark: str
    encoding: VegaZeroEncoding
    data: Optional[str] = None
    transform: Optional[VegaZeroTransform] = None

    _PATTERN_01: ClassVar[re.Pattern] = re.compile(r"mark (\S+) (.+)")
    _PATTERN_02: ClassVar[re.Pattern] = re.compile(r"data (\S+) (.+)")
    _PATTERN_03: ClassVar[re.Pattern] = re.compile(r"encoding (.+)")
    _PATTERN_04: ClassVar[re.Pattern] = re.compile(r"(.+) transform (.+)")

    @classmethod
    def parse(cls, vega_zero_str: str) -> "VegaZero":
        match = cls._PATTERN_01.match(vega_zero_str)

        if match is None:
            raise VegaZeroError(f"Failed to parse vega_zero_str: {vega_zero_str}")

        mark, rest = match.groups()

        match = cls._PATTERN_02.match(rest)

        if match is None:
            data = None
        else:
            data, rest = match.groups()

        match = cls._PATTERN_03.match(rest)

        if match is None:
            raise VegaZeroError(f"Failed to parse vega_zero_str: {vega_zero_str}")

        encoding = match.groups()[0]

        match = cls._PATTERN_04.match(encoding)

        if match is None:
            transform = None
        else:
            encoding, transform = match.groups()

            transform = VegaZeroTransform.parse(transform)

        encoding = VegaZeroEncoding.parse(encoding)

        return cls(
            mark=mark,
            data=data,
            encoding=encoding,
            transform=transform,
        )

    def __str__(self):
        vega_zero_str = [
            f"mark {self.mark}",
            f"data {self.data}" if self.data is not None else None,
            f"encoding {self.encoding}",
            f"transform {self.transform}" if self.transform is not None else None,
        ]

        vega_zero_str = " ".join(s for s in vega_zero_str if s is not None)

        return vega_zero_str

    # TODO: Refactor again (Still too long. Split into sub sections)
    def to_vega_lite(self, data: Optional[Union[dict, pd.DataFrame]] = None) -> dict:
        if isinstance(data, pd.DataFrame):
            data = {"values": data.to_dict(orient="records")}

        mark = self.mark

        if mark == "arc":
            if self.encoding.color is None:
                color = self.encoding.x
            else:
                color = self.encoding.color

            encoding = {
                "color": {
                    "field": color,
                    "type": "nominal",
                },
                "theta": {
                    "field": self.encoding.y,
                    "type": "quantitative",
                },
            }

            if self.encoding.y_aggregate is not None:
                encoding["theta"]["aggregate"] = self.encoding.y_aggregate
        else:
            encoding = {
                "x": {
                    "field": self.encoding.x,
                    "type": "quantitative" if mark == "point" else "nominal",
                },
                "y": {
                    "field": self.encoding.y,
                    "type": "quantitative",
                },
            }

            if self.encoding.y_aggregate is not None:
                encoding["y"]["aggregate"] = self.encoding.y_aggregate

            if self.encoding.color is not None:
                encoding["color"] = {"field": self.encoding.color, "type": "nominal"}

        if self.transform is None:
            vega_lite = {
                "mark": mark,
                "encoding": encoding,
            }

            if data is not None:
                vega_lite["data"] = data

            return vega_lite

        if self.transform.bin is not None:
            if mark == "arc":
                raise VegaZeroError(f"mark arc doesn't support transform.bin")

            (axis, time_unit) = self.transform.bin

            if axis != "x":
                raise VegaZeroError(f"Unsupported binning axis: {axis}")

            encoding[axis]["type"] = "temporal"

            if time_unit == "weekday":
                encoding[axis]["timeUnit"] = "week"
            else:
                encoding[axis]["timeUnit"] = time_unit

        transform, transform_filters = [], []

        if self.transform.sort is not None:
            (axis, order) = self.transform.sort

            if mark == "arc":
                encoding["order"] = {
                    "field": "value",
                    "type": "quantitative",
                    "sort": "descending" if order == "desc" else "ascending",
                }
            else:
                # https://github.com/Thanksyy/ncNet/blob/19e852368228ad251a28623950524a364ee95260/utilities/vis_rendering.py#L244
                if axis == "x":
                    # The line below is the original code, but this can't correctly sort x-axis in the alphabet order
                    # encoding["y"]["sort"] = f"-x" if order == "desc" else "x"
                    encoding[axis]["sort"] = (
                        "descending" if order == "desc" else "ascending"
                    )
                else:
                    encoding["x"]["sort"] = f"-y" if order == "desc" else "y"

        if self.transform.filter is not None:
            filter_ = self.transform.filter

            filter_ = re.sub(
                r"(\S+ )?(\S+) between (\S+) and (\S+)",
                r"\1\3 <= \2 & \2 <= \4",
                filter_,
            )

            # TODO: Support other patterns but %(\S+)%, %(\S+) and (\S+)% (e.g. %a%b%c%)

            filter_ = re.sub(
                r'(\S+ )?(\S+) not like "%(\S+)%"',
                r"! \1test( /.*\3.*/g , \2 )",
                filter_,
            )

            filter_ = re.sub(
                r'(\S+ )?(\S+) not like "%(\S+)"', r"! \1test( /.*\3/g , \2 )", filter_
            )

            filter_ = re.sub(
                r'(\S+ )?(\S+) not like "(\S+)%"', r"! \1test( /\3.*/g , \2 )", filter_
            )

            filter_ = re.sub(
                r'(\S+ )?(\S+) like "%(\S+)%"', r"\1test( /.*\3.*/g , \2 )", filter_
            )

            filter_ = re.sub(
                r'(\S+ )?(\S+) like "%(\S+)"', r"\1test( /.*\3/g , \2 )", filter_
            )

            filter_ = re.sub(
                r'(\S+ )?(\S+) like "(\S+)%"', r"\1test( /\3.*/g , \2 )", filter_
            )

            filter_ = filter_.replace(" and ", " & ")
            filter_ = filter_.replace(" or ", " | ")
            filter_ = filter_.replace(" = ", " == ")

            if data is None:
                columns = frozenset()
            else:
                columns = frozenset(data["values"][0].keys())

            filter_ = " ".join(
                f"datum.{token}" if token in columns else token
                for token in filter_.split()
            )

            transform_filters.append(filter_)

        if self.transform.topk is not None:
            if self.transform.sort is None:
                raise VegaZeroError("transform.sort must be set if transform.topk used")

            (sort_axis, sort_order) = self.transform.sort

            if sort_order == "desc":
                sort_order = "descending"
            elif sort_order == "asc":
                sort_order = "ascending"
            else:
                raise VegaZeroError(f"Unsupported sort order: {sort_order}")

            if self.mark == "arc":
                if sort_axis == "x":
                    field = self.encoding.x
                elif sort_axis == "y":
                    field = self.encoding.y
                else:
                    field = sort_axis

            else:
                field = encoding[sort_axis]["field"]

            # https://vega.github.io/vega-lite/examples/window_top_k.html
            transform_filters.append(f"datum.rank <= {self.transform.topk}")

            transform.append(
                {
                    "window": [
                        {
                            "field": field,
                            "op": "dense_rank",
                            "as": "rank",
                        },
                    ],
                    "sort": [
                        {"field": field, "order": sort_order},
                    ],
                },
            )

        if len(transform_filters) > 0:
            transform.append({"filter": " & ".join(transform_filters)})

        if len(transform) == 0:
            vega_lite = {
                "mark": mark,
                "encoding": encoding,
            }
        else:
            vega_lite = {
                "mark": mark,
                "encoding": encoding,
                "transform": transform,
            }

        if data is not None:
            vega_lite["data"] = data

        return vega_lite
