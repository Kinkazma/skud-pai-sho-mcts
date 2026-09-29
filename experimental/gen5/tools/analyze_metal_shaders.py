#!/usr/bin/env python3
"""Summarize xctrace shader samples for ONE process; durations are not GPU utilization."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def summarize(root, pid, start_seconds, end_seconds):
    references = {e.attrib["id"]: e for e in root.iter() if "id" in e.attrib}

    def resolve(element):
        return references[element.attrib["ref"]] if "ref" in element.attrib else element

    totals = defaultdict(lambda: {"samples": 0, "summed_sample_seconds": 0.0})
    start, end = int(start_seconds * 1e9), int(end_seconds * 1e9)
    if end <= start:
        raise ValueError("end must follow start")
    for node in root.findall("node"):
        schema = node.find("schema")
        if schema is None or schema.get("name") != "metal-shader-profiler-intervals":
            continue
        columns = [c.findtext("mnemonic") for c in schema]
        for row in node.findall("row"):
            cells = dict(zip(columns, map(resolve, row)))
            process_pid = cells["process"].find("pid")
            if process_pid is None or int(resolve(process_pid).text) != pid:
                continue
            timestamp = int(cells["start"].text)
            duration = int(cells["duration"].text)
            clipped = max(0, min(end, timestamp + duration) - max(start, timestamp))
            if not clipped:
                continue
            name = re.sub(r" \(\d+\)$", "", cells["name"].text or cells["name"].get("fmt", ""))
            totals[name]["samples"] += 1
            totals[name]["summed_sample_seconds"] += clipped / 1e9
    total = sum(t["summed_sample_seconds"] for t in totals.values())
    shaders = [{"name": name, **values,
                "share_of_summed_sample_duration_percent": 100 * values["summed_sample_seconds"] / total}
               for name, values in totals.items()] if total else []
    return {"pid": pid, "window_seconds": [start_seconds, end_seconds],
            "scope": "sum of sampled shader intervals; overlaps possible; NOT wall time or core utilization",
            "summed_sample_seconds": total,
            "shaders": sorted(shaders, key=lambda s: -s["summed_sample_seconds"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start", type=float, default=6)
    parser.add_argument("--end", type=float, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(ET.parse(args.xml).getroot(), args.pid, args.start, args.end)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as destination:
            destination.write(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
