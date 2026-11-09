"""Reprodução progressiva ou acompanhamento de JSONL, com alertas antes do fim do fluxo."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from src.anomalies.engine import AnomalyEngine, load_events
from src.reporting.alert_store import AlertStore


def run_events(events, store, rules=None, interval=0., run_id=None):
    engine = AnomalyEngine(rules)
    run_id = run_id or uuid.uuid4().hex
    summary = {"run_id": run_id, "records": 0, "alerts": [], "suppressed": 0,
               "mode": "progressive_replay", "rules": engine.rules}
    for event in events:
        started = time.perf_counter()
        alerts = engine.process(event)
        summary["records"] += 1
        for alert in alerts:
            delivered = store.publish(alert, run_id, started)
            if delivered:
                summary["alerts"].append(delivered)
                print(json.dumps(delivered, ensure_ascii=False), flush=True)
            else:
                summary["suppressed"] += 1
        if interval:
            time.sleep(interval)
    summary["data_quality"] = engine.quality
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def follow_events(path):
    # Processa linhas completas; uma linha sendo escrita não é consumida pela metade.
    with Path(path).open(encoding="utf-8") as stream:
        pending = ""
        while True:
            chunk = stream.readline()
            if not chunk:
                time.sleep(.2)
                continue
            pending += chunk
            if pending.endswith("\n"):
                if pending.strip():
                    yield json.loads(pending)
                pending = ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True)
    parser.add_argument("--rules")
    parser.add_argument("--store", default="outputs/alerts.sqlite")
    parser.add_argument("--output", default="outputs/monitor_summary.json")
    parser.add_argument("--interval", type=float, default=.2)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args(argv)
    if args.interval < 0:
        parser.error("interval deve ser positivo ou zero")
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8")) if args.rules else None
    try:
        summary = run_events(follow_events(args.events) if args.follow else load_events(args.events),
                             AlertStore(args.store), rules, args.interval)
    except KeyboardInterrupt:
        print("Monitor interrompido; alertas já entregues permanecem no receptor.")
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"Entrada inválida: {exc}")
        return 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
