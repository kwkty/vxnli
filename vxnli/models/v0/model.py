import warnings

from pathlib import Path
from typing import Union

import pandas as pd

from transformers import (
    BartForConditionalGeneration,
    TapexTokenizer,
)

from vxnli.errors import InputError


class Model:
    def __init__(self, huggingface_model: Union[str, Path] = "kwkty/vxnli-v0") -> None:
        self.tokenizer = TapexTokenizer.from_pretrained(huggingface_model)
        self.model = BartForConditionalGeneration.from_pretrained(huggingface_model)

    def __call__(self, table: pd.DataFrame, *args, **kwargs) -> str:
        args = args[1:]

        if len(kwargs) > 0 or len(args) != 1 or not isinstance(args[0], str):
            raise InputError("model v0 supports single nl query only")

        query = self._preprocess_args(*args, **kwargs)

        table = table.copy()
        table = self._preprocess_table(table)

        encoding = self.tokenizer(
            table=table,
            query=query,
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with warnings.catch_warnings():
            # Disable warning below
            # UserWarning: Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to 1024 (`self.config.max_length`). Controlling `max_length` via the config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.
            warnings.filterwarnings("ignore", category=UserWarning)
            output = self.model.generate(**encoding)

        output = self.tokenizer.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        output = output[0]

        # Tokenizer might use add_prefix_space=True
        output = output.strip()

        return output

    @staticmethod
    def _preprocess_args(*args, **kwargs) -> str:
        return args[0].lower()

    @staticmethod
    def _preprocess_table(table: pd.DataFrame) -> pd.DataFrame:
        table = table.rename(columns={col: col.lower() for col in table.columns})

        for col_name, col_dtype in zip(table.columns, table.dtypes):
            if pd.api.types.is_string_dtype(col_dtype):
                table[col_name] = table[col_name].str.lower()

        # The TAPEX tokenizer raises an error when the table contains non-str columns
        table = table.astype(str)

        return table
