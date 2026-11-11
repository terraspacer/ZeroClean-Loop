"""Dirty-only v6 fallback grammar tests; no benchmark fixtures."""
import unittest
import numpy as np
from zeroclean.data import DirtyTable
from experimental.unified_loop_v6.detection import DetectionConfig, gate_detection


def table(values, column):
    return DirtyTable('fictional', (column,), tuple((v,) for v in values), 'v6-fictional')


class V6FallbackStructuralTests(unittest.TestCase):
    def run_gate(self, values, column='serial', metadata=None):
        t = table(values, column)
        shape=t.shape
        source=np.full(shape, .1)
        inherited=np.full(shape, .1)
        rules=np.zeros(shape,dtype=bool)
        return gate_detection(t, source, inherited, rules, metadata or {
            'classifier_columns': {column: {'source_empty_predicate_fallback': True}}
        }, DetectionConfig(min_domain_observations=3, fallback_min_observations=3))

    def test_fallback_serial_grammar_promotes_outlier(self):
        result=self.run_gate(['1234-567X','1234-567X','1234-567X','1234-567X','bad'])
        self.assertTrue(result['mask'][4,0])
        self.assertEqual(result['diagnostics']['v6_fallback_structural_promotion']['promoted_count'],1)

    def test_fallback_complete_attribute_promotes_missing(self):
        result=self.run_gate(['A1','A2','A3','A4',''], column='attribute')
        self.assertTrue(result['mask'][4,0])

    def test_open_domain_date_missing_is_not_promoted(self):
        result=self.run_gate(['2020-01-01','2021-01-01','2022-01-01',''], column='release_date')
        self.assertFalse(result['mask'][3,0])


if __name__ == '__main__':
    unittest.main()
