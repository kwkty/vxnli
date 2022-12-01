import pandas as pd
import pytest

from vxnli._vega_zero import VegaZero
from vxnli.errors import InputError
from vxnli.models.v0.model import Model


@pytest.fixture(scope="module")
def model():
    return Model()


@pytest.fixture(scope="module")
def table_example_01():
    return pd.DataFrame(
        {
            "School_id": {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"},
            "School_name": {
                0: "Bremen",
                1: "Culver Community",
                2: "Glenn",
                3: "Jimtown",
                4: "Knox Community",
            },
            "Location": {
                0: "Bremen",
                1: "Culver",
                2: "Walkerton",
                3: "Elkhart",
                4: "Knox",
            },
            "Mascot": {
                0: "Lions",
                1: "Cavaliers",
                2: "Falcons",
                3: "Jimmies",
                4: "Redskins",
            },
            "Enrollment": {0: 495, 1: 287, 2: 605, 3: 601, 4: 620},
            "IHSAA_Class": {0: "AA", 1: "A", 2: "AAA", 3: "AAA", 4: "AAA"},
            "IHSAA_Football_Class": {0: "AA", 1: "A", 2: "AAA", 3: "AAA", 4: "AAA"},
            "County": {
                0: "50 Marshall",
                1: "50 Marshall",
                2: "71 St. Joseph",
                3: "20 Elkhart",
                4: "75 Starke",
            },
        }
    )


def test_call(model, table_example_01):
    vega_zero = model(
        table_example_01,
        "plot the number of schools and total enrollment in each county with a scatter chart.",
    )

    # Check if the model generates vega_zero
    VegaZero.from_str(vega_zero)


def test_call_validation(model, table_example_01):
    with pytest.raises(InputError):
        model(
            table_example_01,
            "plot the number of schools and total enrollment in each county.",
            chart="scatter",
        )
