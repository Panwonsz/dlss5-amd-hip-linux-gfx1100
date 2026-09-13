"""Validate actual saved graph logs; no fabricated execution evidence."""
import json
import pathlib
import re
import os

root = pathlib.Path(__file__).resolve().parents[2]
d = root / os.environ.get("DLSS5_GRAPH_TEST_BUILD", "linux/build/graph-audit")
lines = (d / "network-final.log").read_text().splitlines()
rows = [dict(zip(("block", "count", "n", "nonfinite"), map(int, m.groups())))
        for line in lines
        if (m := re.search(r"block=(\d+) count=(\d+) n=(\d+) nonfinite=(\d+)", line))]
assert len(rows) == 142, len(rows)
for start in (0, 71):
    run = rows[start:start + 71]
    assert sorted(r["block"] for r in run) == list(range(71))
    assert all(r["count"] == 1 and r["n"] > 0 and r["nonfinite"] == 0 for r in run)
blocks = (d / "test_graph_blocks.log").read_text().splitlines()
assert len(blocks) == 40
assert all("mismatches=0 max_error=0" in b for b in blocks)
summary = {"traced_runs": 2, "blocks_each": 71, "nonfinite_total": 0,
           "independent_complete_swin_cases": len(blocks),
           "execution": [line for line in lines if not line.startswith("graph block=")],
           "vit": (d / "test_graph_vit.log").read_text().strip(),
           "ops": (d / "test_graph_ops.log").read_text().strip(),
           "stream": (d / "test_graph_stream.log").read_text().strip()}
(d / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
