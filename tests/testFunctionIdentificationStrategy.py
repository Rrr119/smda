import unittest

from smda.common.FunctionIdentificationStrategy import (
    BASELINE,
    EVIDENCE,
    FunctionIdentificationStrategy,
    resolveFunctionIdentificationStrategy,
)
from smda.Disassembler import Disassembler
from smda.SmdaConfig import SmdaConfig


class FunctionIdentificationStrategyTestSuite(unittest.TestCase):
    def test_baseline_is_the_default(self):
        strategy = resolveFunctionIdentificationStrategy()
        self.assertEqual(strategy.name, BASELINE)
        self.assertFalse(strategy.split_absorbed_tailcalls)
        self.assertTrue(strategy.merge_tailcall_edges)

    def test_evidence_enables_tailcall_boundary_recovery(self):
        strategy = FunctionIdentificationStrategy(EVIDENCE)
        self.assertTrue(strategy.split_absorbed_tailcalls)
        self.assertEqual(strategy.name, EVIDENCE)

    def test_aliases_and_existing_instances_resolve(self):
        self.assertEqual(FunctionIdentificationStrategy("default"), FunctionIdentificationStrategy(BASELINE))
        self.assertEqual(FunctionIdentificationStrategy("tailcalls"), FunctionIdentificationStrategy(EVIDENCE))
        strategy = FunctionIdentificationStrategy(EVIDENCE)
        self.assertIs(resolveFunctionIdentificationStrategy(strategy), strategy)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            FunctionIdentificationStrategy("everything")

    def test_config_is_normalized_by_the_disassembler(self):
        config = SmdaConfig()
        config.FUNCTION_IDENTIFICATION_STRATEGY = "evidence"
        disassembler = Disassembler(config, backend="intel")
        self.assertIsInstance(config.FUNCTION_IDENTIFICATION_STRATEGY, FunctionIdentificationStrategy)
        self.assertEqual(config.FUNCTION_IDENTIFICATION_STRATEGY.name, EVIDENCE)
        self.assertIsNotNone(disassembler)

    def test_strategy_serializes(self):
        self.assertEqual(
            FunctionIdentificationStrategy(EVIDENCE).toDict(),
            {
                "name": EVIDENCE,
                "split_absorbed_tailcalls": True,
                "merge_tailcall_edges": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
