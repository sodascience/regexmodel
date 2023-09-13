# regexmodel

Regexmodel is a python package that uses a graph model to fit and synthesize structured strings.
Structured strings are strings such as license plates, credit card numbers ip-addresses, and phone numbers. Regexmodel can infer a regex-like structure from a series of positive examples and create new samples
(such as phone numbers etc.).

Features:

- Draw new synthetic values
- Only on the numpy and polar libraries (faker for benchmarks).
- Fast (on average < 1 second for about 500 positive examples).
- Can provide statistics on how good the regexmodel has fit your values using log likelihood.
- Can be serialized and can be modified by hand.

## Installation

You can install regexmodel using pip:

```bash
pip install regexmodel
```

If you want the latest version of git, use:

```bash
pip install git+https://github.com/sodascience/regexmodel.git
```

If you want to run the benchmarks, you should also install the faker package:

```bash
pip install faker
```

## Using regexmodel

Fitting the regexmodel is as simple as:

```python
from regexmodel import RegexModel

model = RegexModel.fit(your_values_to_fit, count_thres=10, method="accurate")
```

The `count_thres` parameter changes how detailed and time consuming the fit is. A higher threshold means
a shorter time to fit, but also a worse fit.

The `method` parameter determines the performance/how fast the model is trained. For better looking results,
the "accurate" method is advised. If the quickness of the fit is more important, then you can use the "fast" method. The "accurate" method is generally slow with very long/branching/unstructured strings.

Then synthesizing a new value is done with:

```python
model.draw()
```

## Serialization

The regex model can be serialized so that it can be stored in for example a JSON file:

```python
import json
with open(some_file, "w") as handle:
    json.dump(model.serialize(), handle)
```

And deserialized:

```python
with open(some_file, "r") as handle:
    model = RegexModel(json.load(handle))
```
