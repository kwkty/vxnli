# Extended Natural Language Interface for Data Visualization (V-XNLI)

> **Warning**
> This is a research project. I don't consider real-world usage.

**[DEMO](https://colab.research.google.com/drive/1kithG8Hy-cCQNJX8HBRrEwd3uXLcKiA2?usp=sharing)**

Do you remember the matplotlib APIs? I don't.
It's troublesome for novice users to learn them.
Especially, visualization tools have many options.

In recent years, Natural Language Interface for Data Visualization (V-NLI) gets popular,
and some commercial tools like Tableau and Power BI have already implemented it.
With V-NLI, users can directly tell their intent to the system through natural language,
so there is no need to learn the complicated WIMP operations.
For example, if you want to show the top 5 confirmed cases by state with a bar chart in COVID-19 data analysis,
you have only to say "show the top 5 confirmed cases by state with a bar chart".

Thanks to deep learning, the language models have improved their prediction performance,
and many researchers focus on interpreting more complex and ambiguous input.
However, in my view, it is also essential to simplify the user's input.

Therefore, I propose an extended method of V-NLI, Extended Natural Language Interface for Data Visualization (V-XNLI).
It can accept not only single text input but also multiple text and key-value inputs.
You can set any value to both keys and values, so you don't have to remember how to use it, just like NLI.

I implemented it with Python's variable-length arguments.
The definitions of V-NLI and V-XNLI are the following.

```python
def vnli(text: str) -> Figure:
    ...

def vxnli(*args, **kwargs) -> FIgure:
    ...
```

V-XNLI can take single text input, so it is a superset of V-NLI.
Besides, you can consider V-XNLI as matplotlib which you don't have to remember its APIs.
For example, if you want to show the top 5 confirmed cases by state with a bar chart in COVID-19 data analysis, then

```python
vxnli("show the top 5 confirmed cases by state with a bar chart")

# Or
vxnli("show the top 5 confirmed cases by state", chart="bar")

# Or
vxnli(x="state", y="confirmed cases", limit=5, chart="bar")
```

Like this, XNLI can replace some or all text with structured input, or simplify user's input.

## Installation

```bash
pip install git+https://github.com/kwkty/vxnli@0.2.0#egg=vxnli[model-v1] 
```

## Usage

```python
import pandas as pd

from vxnli import Plot

df = pd.read_csv("path/to/data")

plot = Plot()

plot(df, "... scatterplot ...")

plot("...", "... scatter chart ...", df, ...)

plot({"figure": "scatter"}, df, "...", True, ...)

plot("...", ..., data=df, graph="scatter", foo=1, bar=None, ...)

plot(foo={...}, use_scatter_plot=True, bar=(...), df=df, ...)

plot("... not directly", d=df, but="semantically", specify=["it", "..."])
```

## Development

```bash
# Install nvBench dataset
git clone https://github.com/TsinghuaDatabaseGroup/nvBench data/datasets/nvBench
unzip data/datasets/nvBench/databases.zip -d data/datasets/nvBench/

# Install nvBench dataset (preprocessed by the ncNet authors)
git clone https://github.com/Thanksyy/ncNet data/datasets/ncNet

# Install packages
poetry install
```
