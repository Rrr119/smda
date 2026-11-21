"""Function-level call graph built from a finished :class:`SmdaReport`.

Edges carry every site that produced them, so repeated calls and tail calls
between the same pair of functions become one graph edge rather than parallel
duplicates. Tail calls are explicit because they are control-flow edges between
functions that no ``call`` instruction records.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CALL = "call"
TAIL_CALL = "tail_call"

_CALL_MNEMONICS = {
    "intel": frozenset({"call", "lcall"}),
    "aarch64": frozenset({"bl", "blr"}),
    "arm": frozenset({"bl", "blx"}),
    "cil": frozenset({"call", "calli", "callvirt", "jmp"}),
    "dalvik": frozenset(),
}
_TAIL_BRANCHES = {
    "intel": frozenset({"jmp", "ljmp"}),
    "aarch64": frozenset({"b"}),
    "arm": frozenset({"b"}),
    "cil": frozenset({"br", "jmp"}),
    "dalvik": frozenset({"goto", "goto/16", "goto/32"}),
}


def _baseMnemonic(mnemonic):
    return (mnemonic or "").strip().split(" ")[-1]


def _isDalvikCall(mnemonic):
    return (mnemonic or "").startswith("invoke-")


@dataclass(frozen=True)
class CallSite:
    instruction: int
    kind: str

    def toDict(self):
        return {"instruction": self.instruction, "kind": self.kind}


@dataclass
class CallGraphNode:
    offset: int
    function_name: str = ""
    num_blocks: int = 0
    num_instructions: int = 0
    confidence: Optional[float] = None
    characteristics: Optional[str] = None
    is_exported: bool = False
    is_tailcall_function: bool = False

    def toDict(self):
        return {
            "offset": self.offset,
            "function_name": self.function_name,
            "num_blocks": self.num_blocks,
            "num_instructions": self.num_instructions,
            "confidence": self.confidence,
            "characteristics": self.characteristics,
            "is_exported": self.is_exported,
            "is_tailcall_function": self.is_tailcall_function,
        }


@dataclass
class CallGraphEdge:
    source: int
    target: int
    sites: List[CallSite] = field(default_factory=list)

    @property
    def kinds(self):
        return sorted({site.kind for site in self.sites})

    @property
    def kind(self):
        kinds = self.kinds
        return kinds[0] if len(kinds) == 1 else "mixed"

    def addSite(self, instruction, kind):
        site = CallSite(instruction, kind)
        if site not in self.sites:
            self.sites.append(site)
            self.sites.sort(key=lambda entry: (entry.instruction, entry.kind))

    def toDict(self):
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "kinds": self.kinds,
            "call_sites": [site.toDict() for site in self.sites],
        }


class CallGraph:
    def __init__(self):
        self.nodes: Dict[int, CallGraphNode] = {}
        self.edges: Dict[Tuple[int, int], CallGraphEdge] = {}

    @classmethod
    def fromReport(cls, report, merge_tailcalls=True, include_tailcall_chains=False):
        graph = cls()
        functions = {function.offset: function for function in report.getFunctions()}
        for offset, function in sorted(functions.items()):
            graph.addNode(
                CallGraphNode(
                    offset=offset,
                    function_name=function.function_name or "",
                    num_blocks=function.num_blocks,
                    num_instructions=function.num_instructions,
                    confidence=function.confidence,
                    characteristics=function.characteristics,
                    is_exported=function.is_exported,
                    is_tailcall_function=bool(function.characteristics and "t" in function.characteristics),
                )
            )
        architecture = report.architecture or "intel"
        call_mnemonics = _CALL_MNEMONICS.get(architecture, frozenset())
        tail_branches = _TAIL_BRANCHES.get(architecture, frozenset())
        for source_offset, function in sorted(functions.items()):
            for block in function.getBlocks():
                instructions = list(block.getInstructions())
                for index, instruction in enumerate(instructions):
                    mnemonic = _baseMnemonic(instruction.mnemonic)
                    kind = None
                    if _isDalvikCall(instruction.mnemonic) or mnemonic in call_mnemonics:
                        kind = CALL
                    elif mnemonic in tail_branches and index == len(instructions) - 1:
                        kind = TAIL_CALL
                    if kind is None:
                        continue
                    for target in function.outrefs.get(instruction.offset, ()):
                        target_function = graph._resolveFunction(report, functions, target)
                        if target_function is None:
                            continue
                        graph.addEdge(source_offset, target_function, instruction.offset, kind)
        if merge_tailcalls:
            graph.mergeTailcallEdges()
        if include_tailcall_chains:
            graph.mergeTailcallChains()
        return graph

    @staticmethod
    def _resolveFunction(report, functions, target):
        if target in functions:
            return target
        function = report.findFunctionByContainedAddress(target)
        if function is not None and function.offset in functions:
            return function.offset
        return None

    def addNode(self, node):
        self.nodes[node.offset] = node

    def addEdge(self, source, target, instruction, kind):
        if source not in self.nodes or target not in self.nodes:
            return
        edge = self.edges.get((source, target))
        if edge is None:
            edge = CallGraphEdge(source, target)
            self.edges[(source, target)] = edge
        edge.addSite(instruction, kind)

    def mergeTailcallEdges(self):
        merged = {}
        for edge in self.edges.values():
            key = (edge.source, edge.target)
            existing = merged.get(key)
            if existing is None:
                merged[key] = edge
                continue
            for site in edge.sites:
                existing.addSite(site.instruction, site.kind)
        self.edges = merged
        return self

    def mergeTailcallChains(self):
        outgoing = {}
        for edge in self.edges.values():
            outgoing.setdefault(edge.source, []).append(edge)
        redirected = {}
        for (source, target), edge in list(self.edges.items()):
            if edge.kind != TAIL_CALL:
                continue
            seen = {source}
            current = target
            while current not in seen:
                seen.add(current)
                targets = outgoing.get(current, [])
                if len(targets) != 1 or targets[0].kind != TAIL_CALL:
                    break
                current = targets[0].target
            if current != target:
                redirected[(source, target)] = ((source, current), edge)
        for old_key, (new_key, edge) in redirected.items():
            del self.edges[old_key]
            existing = self.edges.get(new_key)
            if existing is None:
                self.edges[new_key] = CallGraphEdge(new_key[0], new_key[1], list(edge.sites))
            else:
                for site in edge.sites:
                    existing.addSite(site.instruction, site.kind)
        return self

    def getNode(self, offset):
        return self.nodes.get(offset)

    def getEdge(self, source, target):
        return self.edges.get((source, target))

    def getOutgoing(self, source):
        return sorted(
            (edge for edge in self.edges.values() if edge.source == source),
            key=lambda edge: edge.target,
        )

    def getIncoming(self, target):
        return sorted(
            (edge for edge in self.edges.values() if edge.target == target),
            key=lambda edge: edge.source,
        )

    def toDict(self):
        return {
            "directed": True,
            "nodes": [node.toDict() for _, node in sorted(self.nodes.items())],
            "edges": [edge.toDict() for _, edge in sorted(self.edges.items())],
            "statistics": {
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "num_call_sites": sum(len(edge.sites) for edge in self.edges.values()),
            },
        }

    def toJSON(self, indent=None, sort_keys=False):
        return json.dumps(self.toDict(), indent=indent, sort_keys=sort_keys)

    def toDot(self):
        lines = ["digraph callgraph {"]
        for offset, node in sorted(self.nodes.items()):
            label = node.function_name or f"0x{offset:x}"
            label = label.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'  "0x{offset:x}" [label="{label}\\n0x{offset:x}"];')
        for edge in sorted(self.edges.values(), key=lambda entry: (entry.source, entry.target)):
            style = "dashed" if edge.kind == TAIL_CALL else "solid"
            lines.append(f'  "0x{edge.source:x}" -> "0x{edge.target:x}" [label="{edge.kind}", style={style}];')
        lines.append("}")
        return "\n".join(lines)

    def write(self, file_path, output_format=None):
        if output_format is None:
            output_format = "dot" if str(file_path).lower().endswith(".dot") else "json"
        output_format = output_format.lower()
        if output_format == "json":
            content = self.toJSON(indent=2, sort_keys=True)
        elif output_format == "dot":
            content = self.toDot()
        else:
            raise ValueError(f"unsupported call graph format {output_format!r}")
        with open(file_path, "w", encoding="utf-8") as output:
            output.write(content)
