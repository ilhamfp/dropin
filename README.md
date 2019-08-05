# Dropin

Dropin was my attempt to learn about custom layer in keras. Reversing how Dropout work, instead of randomly disable, we randomly enable each neuron on a layer.

## Performance
Full detail available on [`mnist-benchmark.ipynb`](https://github.com/ilhamfp/dropin/blob/master/mnist-benchmark.ipynb) notebook. The Dropin code is available on [`dropin.py`](https://github.com/ilhamfp/dropin/blob/master/dropin.py) or [this kaggle utility script](https://www.kaggle.com/ilhamfp31/dropin).

|  | Original | With Dropout (p=0.25) | With Dropin (p=0.75) |
| ------ | ------ | ------ | ------ |
| **Val Loss** | 0.04478 | 0.02905 | 0.02720 |
| **Val Accuracy** | 0.99080 | 0.99230 | 0.99360 |
