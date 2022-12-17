from typing import Optional

import pytest

from vxnli._vega_zero import VegaZero, VegaZeroEncoding, VegaZeroTransform


@pytest.mark.parametrize(
    "x, y, y_aggregate, color",
    [
        ("product_name", "product_name", "count", None),
        ("product_name", "product_name", "count", "red"),
        ("product_name", "product_name", None, None),
        ("product_name", "product_name", None, "red"),
    ],
)
def test_vega_zero_encoding(
    x: str, y: str, y_aggregate: Optional[str], color: Optional[str]
):
    vega_zero_encoding_str = "none" if y_aggregate is None else y_aggregate
    vega_zero_encoding_str = f"x {x} y aggregate {vega_zero_encoding_str} {y}"

    if color is not None:
        vega_zero_encoding_str += f" color {color}"

    vega_zero_encoding = VegaZeroEncoding.parse(vega_zero_encoding_str)

    assert vega_zero_encoding.x == x
    assert vega_zero_encoding.y == y
    assert vega_zero_encoding.y_aggregate == y_aggregate
    assert vega_zero_encoding.color == color

    assert str(vega_zero_encoding) == vega_zero_encoding_str


@pytest.mark.parametrize(
    "filter, group, bin, sort, topk",
    [
        (None, "x", None, ("y", "desc"), None),
        (None, "x", None, ("y", "desc"), 10),
    ],
)
def test_vega_zero_transform(filter, group, bin, sort, topk):
    vega_zero_transform_str = zip(
        ["filter", "group", "bin", "sort", "topk"],
        [filter, group, bin and " ".join(bin), sort and " ".join(sort), topk],
    )

    vega_zero_transform_str = (
        f"{k} {v}" for k, v in vega_zero_transform_str if v is not None
    )

    vega_zero_transform_str = " ".join(vega_zero_transform_str)

    vega_zero_transform = VegaZeroTransform.parse(vega_zero_transform_str)

    assert vega_zero_transform.filter == filter
    assert vega_zero_transform.group == group
    assert vega_zero_transform.bin == bin
    assert vega_zero_transform.sort == sort
    assert vega_zero_transform.topk == topk

    assert str(vega_zero_transform) == vega_zero_transform_str


@pytest.mark.parametrize(
    "vega_zero",
    [
        "mark arc encoding x country y aggregate count country transform group x",
        'mark bar encoding x job_id y aggregate mean employee_id transform filter hire_date < "2002-06-21" group x sort y asc',
        "mark bar encoding x other_details y aggregate count other_details transform group x sort monthly_rental desc",
        "mark bar encoding x transaction_type y aggregate sum transaction_amount transform group x",
        'mark bar encoding x job_id y aggregate mean salary transform filter salary between 8000 and 12000 and commission_pct != "null" or department_id != 40 group x sort x asc',
        "mark bar encoding x nationality y aggregate count nationality transform group x",
        "mark bar encoding x meter_300 y aggregate none meter_100 transform sort y asc",
        "mark bar encoding x all_home y aggregate mean school_id transform group x sort y asc",
        "mark bar encoding x outcome_code y aggregate count outcome_code transform group x sort x asc",
        'mark arc encoding x sex y aggregate count sex transform filter rank = "asstprof" group x',
        "mark bar encoding x crs_code y aggregate count crs_code transform group x",
        "mark bar encoding x all_games y aggregate none school_id transform sort y asc",
        "mark bar encoding x school_code y aggregate count distinct dept_address",
        "mark bar encoding x product_category_code y aggregate mean product_price transform group x sort y desc",
        'mark line encoding x hire_date y aggregate none employee_id transform filter salary between 8000 and 12000 and commission_pct != "null" or department_id != 40 sort x desc',
        "mark bar encoding x pettype y aggregate mean weight transform group x",
        "mark bar encoding x nationality y aggregate sum age transform group x sort x asc",
        'mark bar encoding x job_id y aggregate sum manager_id transform filter first_name like "%d%" or first_name like "%s%" group x sort x desc',
        'mark bar encoding x job_id y aggregate sum manager_id transform filter hire_date < "2002-06-21" group x sort y asc',
        "mark point encoding x advisor y aggregate count advisor transform group x",
        "mark bar encoding x nationality y aggregate count nationality transform group x",
        "mark point encoding x team_id y aggregate none all_games_percent transform group acc_road",
        "mark arc encoding x languages y aggregate mean rating transform group x",
        "mark bar encoding x all_home y aggregate none school_id color acc_road transform group x sort x desc",
        'mark bar encoding x job_id y aggregate mean salary transform filter salary between 8000 and 12000 and commission_pct != "null" or department_id != 40 group x sort y desc',
        "mark arc encoding x sex y aggregate mean age transform group x",
        'mark bar encoding x hire_date y aggregate mean department_id transform filter hire_date < "2002-06-21" sort y desc bin x by weekday',
        "mark point encoding x gameid y aggregate sum hours_played transform group x",
        "mark bar encoding x name y aggregate count name transform group x sort x asc",
        'mark bar encoding x number_of_matches y aggregate count number_of_matches transform filter injury != "knee problem" group x sort y asc',
    ],
)
def test_vega_zero_parse(vega_zero: str):
    # TODO: Check the output with the expected one
    VegaZero.parse(vega_zero)


@pytest.mark.parametrize(
    "vega_zero",
    [
        "mark arc encoding x country y aggregate count country transform group x",
        'mark bar encoding x job_id y aggregate mean employee_id transform filter hire_date < "2002-06-21" group x sort y asc',
        "mark bar encoding x other_details y aggregate count other_details transform group x sort monthly_rental desc",
        "mark bar encoding x transaction_type y aggregate sum transaction_amount transform group x",
        'mark bar encoding x job_id y aggregate mean salary transform filter salary between 8000 and 12000 and commission_pct != "null" or department_id != 40 group x sort x asc',
        "mark bar encoding x nationality y aggregate count nationality transform group x",
        "mark bar encoding x meter_300 y aggregate none meter_100 transform sort y asc",
        "mark bar encoding x all_home y aggregate mean school_id transform group x sort y asc",
        "mark bar encoding x outcome_code y aggregate count outcome_code transform group x sort x asc",
        'mark arc encoding x sex y aggregate count sex transform filter rank = "asstprof" group x',
        "mark bar encoding x crs_code y aggregate count crs_code transform group x",
        "mark bar encoding x all_games y aggregate none school_id transform sort y asc",
        "mark bar encoding x school_code y aggregate count distinct dept_address",
        "mark bar encoding x product_category_code y aggregate mean product_price transform group x sort y desc",
        'mark line encoding x hire_date y aggregate none employee_id transform filter salary between 8000 and 12000 and commission_pct != "null" or department_id != 40 sort x desc',
        "mark bar encoding x pettype y aggregate mean weight transform group x",
        "mark bar encoding x nationality y aggregate sum age transform group x sort x asc",
        'mark bar encoding x job_id y aggregate sum manager_id transform filter first_name like "%d%" or first_name like "%s%" group x sort x desc',
        'mark bar encoding x job_id y aggregate sum manager_id transform filter hire_date < "2002-06-21" group x sort y asc',
        "mark point encoding x advisor y aggregate count advisor transform group x",
        "mark bar encoding x nationality y aggregate count nationality transform group x",
        "mark point encoding x team_id y aggregate none all_games_percent transform group acc_road",
        "mark arc encoding x languages y aggregate mean rating transform group x",
        "mark bar encoding x all_home y aggregate none school_id color acc_road transform group x sort x desc",
        'mark bar encoding x job_id y aggregate mean salary transform filter salary between 8000 and 12000 and commission_pct != "null" or department_id != 40 group x sort y desc',
        "mark arc encoding x sex y aggregate mean age transform group x",
        'mark bar encoding x hire_date y aggregate mean department_id transform filter hire_date < "2002-06-21" sort y desc bin x by weekday',
        "mark point encoding x gameid y aggregate sum hours_played transform group x",
        "mark bar encoding x name y aggregate count name transform group x sort x asc",
        'mark bar encoding x number_of_matches y aggregate count number_of_matches transform filter injury != "knee problem" group x sort y asc',
    ],
)
def test_vega_zero_to_vega_lite(vega_zero: str):
    # TODO: Check the output with the expected one
    VegaZero.parse(vega_zero).to_vega_lite()
