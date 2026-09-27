"""Usage: python -m pipeline analyze --video ... --config ... --output ..."""

import argparse
import json

from pipeline.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--video", required=True)
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--roster")
    analyze.add_argument("--max-frames", type=int)
    analyze.add_argument("--mode", choices=("parallel", "sequential"))
    recompute = commands.add_parser("recompute")
    recompute.add_argument("--run", required=True)
    recompute.add_argument("--output", required=True)
    args = parser.parse_args()
    from pipeline.run import analyze as run_analysis, recompute as run_recompute
    if args.command == "analyze":
        cfg = load_config(args.config, {"max_frames": args.max_frames, "mode": args.mode})
        result = run_analysis(args.video, args.output, cfg, args.roster)
    else:
        result = run_recompute(args.run, args.output)
    print(json.dumps({"status": result["status"], "players": len(result["players"]), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
