"""Promote absorbed tailcall targets to function boundaries.

A tailcall is a terminal direct branch from one function to another. When the
callee has no call reference of its own, recursive traversal can decode it as a
successor block of the caller. This pass recognizes boundaries that terminal
branches and placement evidence declare independently of a local prologue, then
partitions the absorbed body into its own function without re-running the whole
analysis.
"""

from bisect import bisect_right

_INTEL_TAIL_BRANCHES = frozenset({"jmp", "ljmp"})
_AARCH64_TAIL_BRANCHES = frozenset({"b"})
_ARM_TAIL_BRANCHES = frozenset({"b"})
_CIL_TAIL_BRANCHES = frozenset({"br", "jmp"})
_DALVIK_TAIL_BRANCHES = frozenset({"goto"})

_TAIL_BRANCHES = {
    "intel": _INTEL_TAIL_BRANCHES,
    "aarch64": _AARCH64_TAIL_BRANCHES,
    "arm": _ARM_TAIL_BRANCHES,
    "cil": _CIL_TAIL_BRANCHES,
    "dalvik": _DALVIK_TAIL_BRANCHES,
}

_ARCHITECTURE_PADDING = {
    "intel": (16, frozenset({0x00, 0x90, 0xCC})),
    "aarch64": (4, frozenset({0x00, 0x1F, 0x20, 0x03, 0xD5})),
    "arm": (4, frozenset({0x00, 0xA0, 0xE1})),
}


class TailcallMerger:
    def __init__(self, disassembler):
        self.disassembler = disassembler

    def merge(self):
        if not self.disassembler.config.FUNCTION_IDENTIFICATION_STRATEGY.split_absorbed_tailcalls:
            return []
        return self.splitAbsorbedTailcallTargets()

    def splitAbsorbedTailcallTargets(self):
        disassembly = self.disassembler.disassembly
        architecture = disassembly.binary_info.architecture
        tail_branches = _TAIL_BRANCHES.get(architecture, frozenset())
        if not tail_branches:
            return []
        split_points_by_owner = {}
        for owner in sorted(disassembly.functions):
            if self.disassembler._analysisTimeoutTripped():
                break
            blocks = disassembly.functions[owner]
            function_addrs = {instruction[0] for block in blocks for instruction in block}
            block_starts = {block[0][0] for block in blocks if block}
            if not function_addrs:
                continue
            for block in blocks:
                for instruction in block:
                    source_addr, source_size, mnemonic = instruction[:3]
                    if self._baseMnemonic(mnemonic) not in tail_branches:
                        continue
                    for target in self._targets(source_addr):
                        if target not in function_addrs or target not in block_starts or target == owner:
                            continue
                        if not self._hasBoundaryEvidence(owner, target, source_addr, source_size):
                            continue
                        if not self._isIndependentTail(owner, target, source_addr):
                            continue
                        split_points_by_owner.setdefault(owner, set()).add(target)
        if self.disassembler._analysisTimeoutTripped():
            return []
        promoted = []
        for owner, split_points in sorted(split_points_by_owner.items()):
            accepted = self._filterSafeSplitPoints(owner, sorted(split_points))
            if not accepted:
                continue
            for target in accepted:
                self._markTailcallTarget(target)
            if self.disassembler._partitionFunctionAtPdataEnds(owner, accepted):
                promoted.extend(accepted)
        return sorted(set(promoted))

    @staticmethod
    def _baseMnemonic(mnemonic):
        return (mnemonic or "").split(" ")[-1]

    def _targets(self, source_addr):
        refs = dict.get(self.disassembler.disassembly.code_refs_from, source_addr)
        if refs is None:
            return ()
        if isinstance(refs, int):
            return (refs,)
        return refs

    def _hasBoundaryEvidence(self, owner, target, source_addr, source_size):
        fc_manager = self.disassembler.fc_manager
        candidate = fc_manager.getFunctionCandidate(target)
        if candidate is not None and candidate.hasEntryEvidence():
            return True
        if fc_manager.hasCommonPrologue(target):
            return True
        return self._startsAfterPadding(owner, target, source_addr + source_size)

    def _startsAfterPadding(self, owner, target, tail_source_end):
        architecture = self.disassembler.backend.name
        alignment, padding_bytes = _ARCHITECTURE_PADDING.get(architecture, (4, frozenset({0x00})))
        if target % alignment:
            return False
        disassembly = self.disassembler.disassembly
        previous_end = max(
            (
                instruction[0] + instruction[1]
                for block in disassembly.functions[owner]
                for instruction in block
                if instruction[0] < target
            ),
            default=tail_source_end,
        )
        if not tail_source_end <= previous_end <= target:
            return False
        gap = disassembly.getBytes(previous_end, target - previous_end) or b""
        return bool(gap) and all(byte in padding_bytes for byte in gap)

    def _isIndependentTail(self, owner, target, source_addr):
        disassembly = self.disassembler.disassembly
        instructions = {instruction[0]: instruction for block in disassembly.functions[owner] for instruction in block}
        ordered = sorted(instructions)
        index = bisect_right(ordered, target) - 1
        if index < 0 or ordered[index] != target:
            return False
        tail_addrs = set(ordered[index:])
        for source in tail_addrs:
            refs = dict.get(disassembly.code_refs_from, source)
            if refs is None:
                continue
            if isinstance(refs, int):
                refs = (refs,)
            for destination in refs:
                if destination < target and destination not in disassembly.functions:
                    return False
        return source_addr in instructions

    def _filterSafeSplitPoints(self, owner, split_points):
        accepted = []
        disassembly = self.disassembler.disassembly
        instructions = sorted(instruction[0] for block in disassembly.functions[owner] for instruction in block)
        for target in split_points:
            index = bisect_right(instructions, target) - 1
            if index < 0 or instructions[index] != target or target <= owner:
                continue
            accepted.append(target)
        return accepted

    def _markTailcallTarget(self, target):
        candidate = self.disassembler.fc_manager.getFunctionCandidate(target)
        if candidate is None:
            self.disassembler.fc_manager.ensureCandidate(target)
            candidate = self.disassembler.fc_manager.getFunctionCandidate(target)
        if candidate is not None:
            candidate.setIsTailcallCandidate(True)
            candidate.setAnalysisCompleted()


def mergeAbsorbedTailcalls(disassembler):
    strategy = disassembler.config.FUNCTION_IDENTIFICATION_STRATEGY
    if getattr(strategy, "split_absorbed_tailcalls", False):
        return TailcallMerger(disassembler).merge()
    return []
