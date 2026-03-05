from typing import Dict, List, Any, Optional
import torch
import numpy as np
from .collation_fn import custom_collation_fn

class BucketBatcher:
    """
    A webdataset-compatible filter that batches samples based on their resolution buckets.
    """
    def __init__(
        self, 
        resolution_to_batch_size: Dict[str, int],
        collation_fn=custom_collation_fn,
        default_batch_size: int = 4
    ):
        self.resolution_to_batch_size = resolution_to_batch_size
        self.collation_fn = collation_fn
        self.default_batch_size = default_batch_size
        self.buckets = {}

    def __call__(self, data):
        for sample in data:
            bucket_key = sample.get("resolution_bucket", "default")
            batch_size = self.resolution_to_batch_size.get(bucket_key, self.default_batch_size)
            
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = []
            
            self.buckets[bucket_key].append(sample)
            
            if len(self.buckets[bucket_key]) >= batch_size:
                batch = self.buckets.pop(bucket_key)
                yield self.collation_fn(batch)

def get_resolution_to_batch_size_map(budgets, base_batch_sizes):
    """
    Generates a map from 'HxW' strings to batch sizes based on budgets.
    """
    ar_list = [0.25, 0.33, 0.5, 0.66, 1.0, 1.5, 2.0, 3.0, 4.0]
    res_map = {}
    for budget, b_size in zip(budgets, base_batch_sizes):
        for ar in ar_list:
            h = np.sqrt(budget / ar)
            w = h * ar
            h = int(round(h / 64) * 64)
            w = int(round(w / 64) * 64)
            h, w = max(64, h), max(64, w)
            res_map[f"{h}x{w}"] = b_size
    return res_map
