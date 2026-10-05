"""Read an ELab experiment's basic metadata without exposing its body or secrets."""
import argparse
import json
import os
from pathlib import Path
import sys


def positive_id(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("experiment ID must be a positive integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("experiment ID must be a positive integer")
    return number


def inspect_experiment(experiment_id):
    # Lazy imports keep --help usable without credentials or the optional SDK.
    from tools.elab.cli.elab_cli_simple import build_api_client
    from elabapi_python import ExperimentsApi

    client = build_api_client()
    try:
        experiment = ExperimentsApi(client).get_experiment(id=experiment_id)
        result = {}
        secret = os.environ.get("ELAB_API_KEY", "")
        for field in ("id", "title", "category", "team"):
            value = experiment.get(field) if isinstance(experiment, dict) else getattr(experiment, field, None)
            # Never serialize nested SDK objects, which can contain private fields.
            if value is not None and type(value) not in (str, int):
                value = None
            if isinstance(value, str) and secret:
                value = value.replace(secret, "[redacted]")
            result[field] = value
        return result
    finally:
        # Older generated SDK clients expose only destructor-based cleanup.
        close = getattr(client, "close", None)
        if callable(close):
            close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True, type=positive_id)
    args = parser.parse_args(argv)
    try:
        result = inspect_experiment(args.experiment_id)
    except Exception:
        print("Unable to inspect experiment. Check credentials, permissions, and connectivity.", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    raise SystemExit(main())
