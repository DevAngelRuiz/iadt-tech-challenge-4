"""Gera telemetria e prescrições sintéticas com controles normais e anômalos."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "data" / "samples" / "monitoring"


def build_case(anomalous=False):
    records = []
    start = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    pid = "sintetica_anomala" if anomalous else "sintetica_normal"
    for tick in range(12):
        stamp = (start + timedelta(seconds=10 * tick)).isoformat()
        bad = anomalous and tick >= 7
        records.append({"kind": "vitals", "patient_id": pid, "timestamp": stamp,
                        "metrics": {"heart_rate_bpm": 140 if bad else 72 + tick % 3,
                                    "spo2_pct": 89 if bad else 98,
                                    "systolic_mmhg": 118 + tick % 3, "diastolic_mmhg": 76 + tick % 2}})
        records.append({"kind": "movement", "patient_id": pid, "timestamp": stamp,
                        "activity_score": .01 if anomalous and tick >= 6 else .35 + .02 * (tick % 3),
                        "patient_present": True, "quality_good": True})
        if tick in {0, 7}:
            records.append({"kind": "prescription", "patient_id": pid, "timestamp": stamp,
                            "medication": "medicamento_demo", "dose": 100 if bad else 25,
                            "unit": "mg", "frequency_per_day": 1,
                            "planned_change": tick == 0 or not anomalous,
                            "reason": "Prescrição inicial ou revisão planejada" if tick == 0 or not anomalous else "",
                            "action": "set"})
    return sorted(records, key=lambda row: row["timestamp"])


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, bad in [("normal", False), ("anomalo", True)]:
        records = build_case(bad)
        (TARGET / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        for kind, filename in [("vitals", "sinais_vitais"), ("prescription", "prescricoes"), ("movement", "movimentacao")]:
            rows = [dict(r) for r in records if r["kind"] == kind]
            for row in rows:
                row.pop("kind")
                if "metrics" in row:
                    row.update(row.pop("metrics"))
            with (TARGET / f"{filename}_{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f"Dados sintéticos gravados em {TARGET}")


if __name__ == "__main__":
    main()
