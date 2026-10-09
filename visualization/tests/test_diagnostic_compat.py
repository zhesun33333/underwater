"""Existing generic metrics plots must retain the current taxonomy's last class."""
import unittest

from testsite.reporting.visualize import _mapped_per_class, _match_l3_key


class TaxonomyCompatibilityTests(unittest.TestCase):
    def test_current_and_legacy_underwater_names_are_retained(self):
        for name in ("underwater_target", "Underwater vehicle", "Underwater target"):
            self.assertEqual(_match_l3_key(name), "underwater_target")
        metric = {"precision": 0.4, "recall": 0.6, "f1": 0.48}
        mapped = _mapped_per_class({"l3_per_class": {"Underwater vehicle": metric}})
        self.assertEqual(mapped, {"underwater_target": metric})


if __name__ == "__main__":
    unittest.main()
