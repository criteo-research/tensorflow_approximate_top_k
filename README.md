# Approximate Top-K operation for TensorFlow
The operation allows to use approximate nearest neighbors search to do faster top-k retrievals over a set of embeddings with the assumption that the underlying embeddings do not change very quickly.

Installation
------------
```bash
$ git clone https://github.com/Ivanopolo/tensorflow_approximate_top_k
$ cd tensorflow_approximate_top_k
$ cmake .
$ make
```

Example
------------
This will index `all_embs` 2-D Tensor and will query it with `target_embs` retrieving 10 (`=num_negative_samples`) closest embeddings. With parameters `num_trees` and `num_iters_per_update` we can control the quality of our approximation vs running time.
```python
import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

lib_path = "~/tensorflow_approximate_top_k/approximate_top_k"
lib = tf.load_op_library(lib_path)

dim = 5
all_embs = rng.rand(10, dim).astype(np.float32)
target_embs = rng.rand(2, dim).astype(np.float32)
sample_ids = sampling.approximate_top_k(
    all_embs, 
    target_embs,
    num_negative_samples=10, 
    num_trees=16,
    num_dims=dim,
    num_iters_per_update=100, 
    metric="cosine")
```
