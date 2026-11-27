"""ivf_tune: make an existing Faiss IVF index reach the same recall with less work.

    import ivf_tune
    tuning = ivf_tune.fit(index, sample_queries)        # learns ~4 bytes per list + a probe count
    tuning.save("tuning.npz")
    D, I = ivf_tune.search(index, tuning, queries, k=10)
"""

from .core import Routing, Tuning, exact_neighbors, fit, search

__all__ = ["Routing", "Tuning", "exact_neighbors", "fit", "search"]
__version__ = "0.1.0"
