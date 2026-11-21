"""Deterministic synthetic boundary corpus and exact-start evaluator.

Run directly to reproduce the 100-sample comparison:

    python -m tests.boundary_corpus --output artifacts/boundary_accuracy_comparison.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from smda.Disassembler import Disassembler
from smda.SmdaConfig import SmdaConfig

BASE_ADDR = 0x400000
BITNESS = 64
SAMPLE_COUNT = 100


@dataclass(frozen=True)
class BoundarySample:
    name: str
    buffer: bytes
    base_addr: int
    bitness: int
    truth: frozenset
    tailcall_site: int
    tailcall_target: int


def _rel32(source_end, target):
    return (target - source_end).to_bytes(4, "little", signed=True)


def _function(prologue, value, epilogue):
    return prologue + b"\xb8" + value.to_bytes(4, "little") + epilogue


def buildBoundarySample(index):
    normal_count = 4 if index < 11 else 3
    buffer = bytearray(b"\xcc" * 0x400)

    def put(address, payload):
        offset = address - BASE_ADDR
        buffer[offset : offset + len(payload)] = payload

    main_addr = BASE_ADDR
    ordinary_addrs = [BASE_ADDR + 0x40 + step * 0x20 for step in range(normal_count)]
    tailcaller_addr = BASE_ADDR + 0x200
    tailed_addr = BASE_ADDR + 0x220

    calls = b""
    call_addr = main_addr + 4
    for target in ordinary_addrs + [tailcaller_addr]:
        calls += b"\xe8" + _rel32(call_addr + 5, target)
        call_addr += 5
    put(main_addr, b"\x55\x48\x89\xe5" + calls + b"\x5d\xc3")

    for number, address in enumerate(ordinary_addrs, 1):
        put(address, _function(b"\x55\x48\x89\xe5", number, b"\x5d\xc3"))

    tailcall_site = tailcaller_addr + 5
    put(
        tailcaller_addr,
        b"\xb8" + (0xA0 + index % 16).to_bytes(4, "little") + b"\xe9" + _rel32(tailcall_site + 5, tailed_addr),
    )
    put(tailed_addr, _function(b"", 0x7F - index % 16, b"\xc3"))

    truth = frozenset([main_addr, *ordinary_addrs, tailcaller_addr, tailed_addr])
    return BoundarySample(
        name=f"boundary_{index:03d}",
        buffer=bytes(buffer),
        base_addr=BASE_ADDR,
        bitness=BITNESS,
        truth=truth,
        tailcall_site=tailcall_site,
        tailcall_target=tailed_addr,
    )


def buildBoundaryCorpus(count=SAMPLE_COUNT):
    return [buildBoundarySample(index) for index in range(count)]


def _config(strategy):
    config = SmdaConfig()
    config.FUNCTION_IDENTIFICATION_STRATEGY = strategy
    config.WITH_STRINGS = False
    config.CALCULATE_HASHING = False
    config.CALCULATE_SCC = False
    config.CALCULATE_NESTING = False
    return config


def evaluateBoundaryRecovery(samples, strategy):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    recovered_total = 0
    truth_total = 0
    for sample in samples:
        report = Disassembler(_config(strategy)).disassembleBuffer(
            sample.buffer,
            sample.base_addr,
            bitness=sample.bitness,
            architecture="intel",
        )
        recovered = {function.offset for function in report.getFunctions()}
        true_positives += len(sample.truth & recovered)
        false_positives += len(recovered - sample.truth)
        false_negatives += len(sample.truth - recovered)
        recovered_total += len(recovered)
        truth_total += len(sample.truth)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / truth_total if truth_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": len(samples),
        "truth_functions": truth_total,
        "recovered_functions": recovered_total,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compareBoundaryStrategies(samples=None):
    samples = samples if samples is not None else buildBoundaryCorpus()
    return {
        "corpus": f"tests/boundary_corpus.py ({len(samples)} synthetic x86-64 samples)",
        "metric": "exact function-start match",
        "baseline_default": evaluateBoundaryRecovery(samples, "baseline"),
        "evidence": evaluateBoundaryRecovery(samples, "evidence"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    comparison = compareBoundaryStrategies(buildBoundaryCorpus(args.samples))
    rendered = json.dumps(comparison, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
