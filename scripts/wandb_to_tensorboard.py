"""Convert a local wandb `.wandb` run file to TensorBoard event files.

Usage:
    python scripts/wandb_to_tensorboard.py <path-to-run-*.wandb> [--out DIR]
    # then: tensorboard --logdir <out-dir>

Works fully offline — does not contact the wandb server.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
from torch.utils.tensorboard import SummaryWriter


def iter_records(wandb_file: str):
    ds = datastore.DataStore()
    ds.open_for_scan(wandb_file)
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        yield rec


def history_item_to_value(item):
    try:
        val = json.loads(item.value_json)
    except Exception:
        return None
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def convert(wandb_file: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir)

    n_points = 0
    n_steps = 0
    for rec in iter_records(wandb_file):
        if not rec.HasField("history"):
            continue
        h = rec.history
        step = h.step.num if h.HasField("step") else n_steps
        step_val = None
        metrics = {}
        for item in h.item:
            key = item.key or ".".join(item.nested_key)
            v = history_item_to_value(item)
            if v is None:
                continue
            if key == "_step":
                step_val = int(v)
            metrics[key] = v
        if step_val is not None:
            step = step_val
        for k, v in metrics.items():
            if k.startswith("_"):
                continue
            writer.add_scalar(k, v, global_step=step)
            n_points += 1
        n_steps += 1
    writer.flush()
    writer.close()
    print(f"Wrote {n_points} scalar points across {n_steps} history rows to {out_dir}")
    return n_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wandb_file", help="Path to run-*.wandb file")
    ap.add_argument("--out", default=None, help="Output dir (default: <wandb_file_dir>/tb)")
    args = ap.parse_args()

    wf = str(Path(args.wandb_file).resolve())
    if not os.path.isfile(wf):
        print(f"Not a file: {wf}", file=sys.stderr)
        sys.exit(1)

    out = args.out or str(Path(wf).parent / "tb")
    convert(wf, out)
    print(f"\nRun: tensorboard --logdir {out}")


if __name__ == "__main__":
    main()
