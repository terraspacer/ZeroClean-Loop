"""Focused v6 tests for generic repair and counterfactual policy behavior."""
from __future__ import annotations

import unittest

import numpy as np

from experimental.unified_loop_v6.repair import select_repairs
from experimental.unified_loop_v6.test_repair import FictionalTable, candidate, mask_for


class GenericRepairTests(unittest.TestCase):
    def test_generic_seed_repair_is_allowed_without_schema_roles(self):
        table = FictionalTable(("left", "right"), (("a", "bad"),))
        result = select_repairs(table, mask_for(table, (0, 1)), {(0, 1): [candidate("good")]})
        self.assertEqual(result["repairs"], {(0, 1): "good"})
        self.assertIsNone(result["diagnostics"]["catalog_roles"])

    def test_candidate_pool_does_not_change_detection_mask(self):
        table = FictionalTable(("left", "right"), (("a", "bad"), ("b", "old")))
        mask = mask_for(table, (0, 1))
        result = select_repairs(table, mask, {
            (0, 1): [candidate("good")],
            (1, 1): [candidate("other")],
        })
        self.assertEqual(result["repairs"], {(0, 1): "good"})
        self.assertFalse(result["diagnostics"]["detection_mask_modified"])
        self.assertEqual(result["diagnostics"]["counts"]["candidate_cells_outside_detection"], 1)

    def test_weighted_consistency_loss_abstains(self):
        table = FictionalTable(("left",), (("bad",),))
        item = candidate("worse", delta_features=[-1.0, -1.0, -1.0, -1.0])
        result = select_repairs(table, [[True]], {(0, 0): [item]})
        self.assertEqual(result["repairs"], {})
        review = result["decisions"][0]["candidate_reviews"][0]
        self.assertIn("weighted_consistency_loss", review["rejection_reasons"])

    def test_non_string_candidate_remains_rejected(self):
        table = FictionalTable(("left",), (("bad",),))
        with self.assertRaises(TypeError):
            select_repairs(table, [[True]], {(0, 0): [candidate(12.0)]})


if __name__ == "__main__":
    unittest.main()
