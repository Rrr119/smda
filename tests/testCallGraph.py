import json
import os
import tempfile
import unittest

from smda.common.CallGraph import CALL, TAIL_CALL, CallGraph, CallGraphEdge, CallGraphNode
from smda.Disassembler import Disassembler
from smda.SmdaConfig import SmdaConfig
from tests.boundary_corpus import buildBoundarySample


def _report(sample):
    config = SmdaConfig()
    config.FUNCTION_IDENTIFICATION_STRATEGY = "evidence"
    config.WITH_STRINGS = False
    config.CALCULATE_HASHING = False
    config.CALCULATE_SCC = False
    config.CALCULATE_NESTING = False
    return Disassembler(config).disassembleBuffer(
        sample.buffer,
        sample.base_addr,
        bitness=sample.bitness,
        architecture="intel",
    )


class CallGraphTestSuite(unittest.TestCase):
    def test_report_call_graph_contains_call_and_tailcall_edges(self):
        sample = buildBoundarySample(0)
        report = _report(sample)
        graph = report.getCallGraph()
        self.assertEqual(set(graph.nodes), set(sample.truth))
        self.assertEqual(len(graph.getIncoming(sample.tailcall_target)), 1)
        tailcall = graph.getEdge(sample.tailcall_site - 5, sample.tailcall_target)
        self.assertIsNotNone(tailcall)
        self.assertEqual(tailcall.kind, TAIL_CALL)
        self.assertEqual(tailcall.sites[0].instruction, sample.tailcall_site)
        main = sample.base_addr
        self.assertTrue(all(edge.kind == CALL for edge in graph.getOutgoing(main)))
        self.assertEqual(graph.toDict()["statistics"]["num_edges"], 6)

    def test_repeated_sites_are_merged_into_one_edge(self):
        sample = buildBoundarySample(0)
        config = SmdaConfig()
        config.CALCULATE_HASHING = False
        config.CALCULATE_SCC = False
        config.CALCULATE_NESTING = False
        report = Disassembler(config).disassembleBuffer(
            sample.buffer,
            sample.base_addr,
            bitness=sample.bitness,
            architecture="intel",
        )
        graph = report.getCallGraph()
        caller = next(iter(graph.getOutgoing(sample.base_addr)))
        graph.addEdge(sample.base_addr, caller.target, caller.sites[0].instruction + 1, CALL)
        merged = graph.getEdge(sample.base_addr, caller.target)
        self.assertEqual(len(merged.sites), 2)
        self.assertEqual(
            len(graph.getOutgoing(sample.base_addr)), len({edge.target for edge in graph.getOutgoing(sample.base_addr)})
        )

    def test_dot_and_json_exports(self):
        sample = buildBoundarySample(0)
        report = _report(sample)
        graph = report.getCallGraph()
        dot = graph.toDot()
        self.assertIn("digraph callgraph", dot)
        self.assertIn("style=dashed", dot)
        payload = json.loads(graph.toJSON())
        self.assertTrue(payload["directed"])
        self.assertEqual(payload["statistics"]["num_nodes"], len(sample.truth))
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            report.exportCallGraph(path)
            with open(path, encoding="utf-8") as handle:
                exported = json.load(handle)
        finally:
            os.unlink(path)
        self.assertEqual(exported["statistics"], payload["statistics"])

    def test_tailcall_chain_merging(self):
        graph = CallGraph()
        for offset in (0x1000, 0x1010, 0x1020):
            graph.addNode(CallGraphNode(offset))
        graph.addEdge(0x1000, 0x1010, 0x1000, TAIL_CALL)
        graph.addEdge(0x1010, 0x1020, 0x1010, TAIL_CALL)
        graph.mergeTailcallChains()
        self.assertIsNotNone(graph.getEdge(0x1000, 0x1020))
        self.assertIsNone(graph.getEdge(0x1000, 0x1010))

    def test_edges_keep_each_distinct_call_site(self):
        edge = CallGraphEdge(0x1000, 0x2000)
        edge.addSite(0x1004, CALL)
        edge.addSite(0x1004, CALL)
        edge.addSite(0x1010, TAIL_CALL)
        self.assertEqual(len(edge.sites), 2)
        self.assertEqual(edge.kind, "mixed")


if __name__ == "__main__":
    unittest.main()
