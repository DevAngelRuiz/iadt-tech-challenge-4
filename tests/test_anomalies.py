from datetime import datetime, timedelta, timezone
import json
import pytest
from src.anomalies.engine import AnomalyEngine
from scripts.prepare_monitoring_data import build_case
from src.monitor import run_events
from src.reporting.alert_store import AlertStore


def event(kind, tick=0, **fields):
    return {"kind": kind, "patient_id": "a", "timestamp": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=tick)).isoformat(), **fields}


def test_normal_and_anomalous_controls(tmp_path):
    normal = run_events(build_case(), AlertStore(tmp_path / "normal.sqlite"))
    bad = run_events(build_case(True), AlertStore(tmp_path / "bad.sqlite"))
    assert normal["alerts"] == []
    types = {a["type"] for a in bad["alerts"]}
    assert types == {"anomalia_sinal_vital", "alteracao_prescricao", "imobilidade_prolongada"}
    assert bad["suppressed"] > 0
    assert all(a["evidence"] and a["processing_latency_ms"] >= 0 for a in bad["alerts"])


def test_temporal_outlier_within_limits_and_patient_isolation():
    engine = AnomalyEngine()
    for tick in range(5):
        assert engine.process(event("vitals", tick, metrics={"heart_rate_bpm": 72})) == []
    alerts = engine.process(event("vitals", 6, metrics={"heart_rate_bpm": 90}))
    assert alerts[0]["evidence"]["method"] == "rolling_median_MAD"
    another = event("vitals", 6, metrics={"heart_rate_bpm": 90})
    another["patient_id"] = "b"
    assert engine.process(another) == []


def test_prescription_units_and_planned_changes():
    engine = AnomalyEngine()
    base = {"medication": "demo", "dose": 1000, "unit": "mcg", "frequency_per_day": 1}
    assert not engine.process(event("prescription", **base))
    assert not engine.process(event("prescription", 1, **{**base, "dose": 1, "unit": "mg"}))
    assert not engine.process(event("prescription", 2, **{**base, "dose": 10, "unit": "mg", "planned_change": True, "reason": "revisão"}))
    alerts = engine.process(event("prescription", 3, **{**base, "dose": 1, "unit": "mg"}))
    assert alerts[0]["type"] == "alteracao_prescricao"


def test_absent_patient_and_bad_quality_do_not_imply_immobility():
    engine = AnomalyEngine()
    for tick in range(0, 100, 10):
        assert not engine.process(event("movement", tick, activity_score=0, patient_present=False))
    assert engine.quality


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "texto"])
def test_bad_vital_reading_rejected(value):
    with pytest.raises(ValueError):
        AnomalyEngine().process(event("vitals", metrics={"heart_rate_bpm": value}))


def test_out_of_order_and_missing_timezone_rejected():
    engine = AnomalyEngine()
    engine.process(event("vitals", 10, metrics={"spo2_pct": 98}))
    with pytest.raises(ValueError):
        engine.process(event("vitals", 5, metrics={"spo2_pct": 98}))
    row = event("vitals", metrics={"spo2_pct": 98})
    row["timestamp"] = "2026-01-01T00:00:00"
    with pytest.raises(ValueError):
        AnomalyEngine().process(row)


def test_store_dedup_severity_upgrade_and_ack(tmp_path):
    store = AlertStore(tmp_path / "alerts.sqlite")
    alert = {"type": "x", "severity": "ATENCAO", "patient_id": "a", "timestamp": "2026-01-01T00:00:00Z"}
    first = store.publish(alert, "run")
    assert store.publish(alert, "run") is None
    assert store.publish({**alert, "severity": "CRITICO"}, "run")
    assert store.acknowledge(first["id"])
    assert any(a["acknowledged_at"] for a in store.list())
