import unittest

import pandas as pd

from wwpgd_local_delta.weightwatcher import _analyze_compat


class _ModernWatcher:
    def __init__(self):
        self.received = None

    def analyze(
        self,
        *,
        plot=False,
        randomize=False,
        min_evals=10,
        savefig="ww-img",
        vectors=False,
        start_ids=0,
        ERG=False,
        svd_method="fast",
    ):
        self.received = {
            "plot": plot,
            "randomize": randomize,
            "min_evals": min_evals,
            "savefig": savefig,
            "vectors": vectors,
            "start_ids": start_ids,
            "ERG": ERG,
            "svd_method": svd_method,
        }
        return pd.DataFrame({"name": ["fc1"], "alpha": [2.0]})


class _LegacyWatcher:
    def __init__(self):
        self.received = None

    def analyze(
        self,
        *,
        plot=False,
        randomize=False,
        min_evals=10,
        savefig="ww-img",
        vectors=False,
        start_ids=0,
        detX=False,
        svd_method="fast",
    ):
        self.received = {
            "plot": plot,
            "randomize": randomize,
            "min_evals": min_evals,
            "savefig": savefig,
            "vectors": vectors,
            "start_ids": start_ids,
            "detX": detX,
            "svd_method": svd_method,
        }
        return pd.DataFrame({"name": ["fc1"], "alpha": [2.0]})


class WeightWatcherCompatibilityTests(unittest.TestCase):
    def test_modern_weightwatcher_uses_erg_directly(self):
        watcher = _ModernWatcher()
        details = _analyze_compat(watcher, min_evals=8, svd_method="accurate")
        self.assertFalse(details.empty)
        self.assertEqual(
            watcher.received,
            {
                "plot": False,
                "randomize": False,
                "min_evals": 8,
                "savefig": False,
                "vectors": False,
                "start_ids": 0,
                "ERG": True,
                "svd_method": "accurate",
            },
        )

    def test_legacy_weightwatcher_falls_back_to_detx(self):
        watcher = _LegacyWatcher()
        details = _analyze_compat(watcher, min_evals=6, svd_method="fast")
        self.assertFalse(details.empty)
        self.assertTrue(watcher.received["detX"])
        self.assertEqual(watcher.received["min_evals"], 6)
        self.assertEqual(watcher.received["svd_method"], "fast")


if __name__ == "__main__":
    unittest.main()
