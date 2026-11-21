import unittest

from smda.Disassembler import Disassembler
from smda.SmdaConfig import SmdaConfig
from tests.boundary_corpus import BASE_ADDR, buildBoundarySample


def _config(strategy):
    config = SmdaConfig()
    config.FUNCTION_IDENTIFICATION_STRATEGY = strategy
    config.WITH_STRINGS = False
    config.CALCULATE_HASHING = False
    config.CALCULATE_SCC = False
    config.CALCULATE_NESTING = False
    return config


def _disassemble(buffer, strategy):
    return Disassembler(_config(strategy)).disassembleBuffer(
        buffer,
        BASE_ADDR,
        bitness=64,
        architecture="intel",
    )


class TailcallMergerTestSuite(unittest.TestCase):
    def test_baseline_absorbs_the_tailcalled_body(self):
        sample = buildBoundarySample(0)
        report = _disassemble(sample.buffer, "baseline")
        functions = {function.offset: function for function in report.getFunctions()}
        self.assertNotIn(sample.tailcall_target, functions)
        absorbed = functions[sample.tailcall_site - 5]
        offsets = {instruction.offset for instruction in absorbed.getInstructions()}
        self.assertIn(sample.tailcall_target, offsets)

    def test_evidence_promotes_the_tailcall_target(self):
        sample = buildBoundarySample(0)
        report = _disassemble(sample.buffer, "evidence")
        functions = {function.offset: function for function in report.getFunctions()}
        self.assertIn(sample.tailcall_target, functions)
        tailed = functions[sample.tailcall_target]
        self.assertEqual(
            [instruction.mnemonic for instruction in tailed.getInstructions()],
            ["mov", "ret"],
        )
        caller = functions[sample.tailcall_site - 5]
        self.assertEqual(
            [(instruction.mnemonic, instruction.operands) for instruction in caller.getInstructions()][-1],
            ("jmp", f"0x{sample.tailcall_target:x}"),
        )
        self.assertIn("t", tailed.characteristics)

    def test_evidence_recovers_every_truth_boundary_in_the_corpus(self):
        sample = buildBoundarySample(17)
        report = _disassemble(sample.buffer, "evidence")
        recovered = {function.offset for function in report.getFunctions()}
        self.assertEqual(recovered, set(sample.truth))

    def test_an_unaligned_target_without_entry_evidence_is_not_split(self):
        sample = buildBoundarySample(0)
        buffer = bytearray(sample.buffer)
        source = sample.tailcall_site
        target = sample.tailcall_target + 3
        offset = source - BASE_ADDR
        buffer[offset + 1 : offset + 5] = (target - (source + 5)).to_bytes(4, "little", signed=True)
        body_offset = sample.tailcall_target - BASE_ADDR
        buffer[body_offset : body_offset + 6] = b"\x0f\x1f\x44\x00\x00\xc3"
        report = _disassemble(bytes(buffer), "evidence")
        self.assertNotIn(target, {function.offset for function in report.getFunctions()})

    def test_baseline_strategy_is_a_noop(self):
        sample = buildBoundarySample(0)
        report = _disassemble(sample.buffer, "baseline")
        self.assertNotIn(sample.tailcall_target, {function.offset for function in report.getFunctions()})

    def test_strategy_can_be_selected_after_the_disassembler_is_constructed(self):
        sample = buildBoundarySample(0)
        config = _config("baseline")
        disassembler = Disassembler(config)
        config.FUNCTION_IDENTIFICATION_STRATEGY = "evidence"
        report = disassembler.disassembleBuffer(
            sample.buffer,
            sample.base_addr,
            bitness=sample.bitness,
            architecture="intel",
        )
        self.assertIn(sample.tailcall_target, {function.offset for function in report.getFunctions()})


if __name__ == "__main__":
    unittest.main()
