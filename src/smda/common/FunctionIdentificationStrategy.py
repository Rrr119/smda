"""Function-boundary identification policies for native-code backends.

The baseline policy is the historical SMDA behaviour. The evidence policy keeps
the same candidate sources and adds a post-analysis tailcall pass that promotes
boundaries the recursive traversal absorbed into their caller.
"""

BASELINE = "baseline"
EVIDENCE = "evidence"

VALID_STRATEGIES = (BASELINE, EVIDENCE)

_ALIASES = {
    "default": BASELINE,
    "conservative": BASELINE,
    "tailcall": EVIDENCE,
    "tailcalls": EVIDENCE,
}


class FunctionIdentificationStrategy:
    BASELINE = BASELINE
    EVIDENCE = EVIDENCE
    name = BASELINE
    split_absorbed_tailcalls = False
    merge_tailcall_edges = True

    def __init__(self, name=BASELINE):
        normalized = _ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
        if normalized not in VALID_STRATEGIES:
            raise ValueError(f"unknown function identification strategy {name!r}; expected one of {VALID_STRATEGIES}")
        self.name = normalized
        self.split_absorbed_tailcalls = normalized == EVIDENCE
        self.merge_tailcall_edges = True

    def __repr__(self):
        return f"FunctionIdentificationStrategy({self.name!r})"

    def __eq__(self, other):
        if isinstance(other, FunctionIdentificationStrategy):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == resolveFunctionIdentificationStrategy(other).name
        return NotImplemented

    def toDict(self):
        return {
            "name": self.name,
            "split_absorbed_tailcalls": self.split_absorbed_tailcalls,
            "merge_tailcall_edges": self.merge_tailcall_edges,
        }


def resolveFunctionIdentificationStrategy(strategy=None):
    if isinstance(strategy, FunctionIdentificationStrategy):
        return strategy
    if strategy is None:
        return FunctionIdentificationStrategy()
    return FunctionIdentificationStrategy(strategy)
