"""Detectores online: limites configurados, mediana/MAD e mudanças de prescrição."""
from collections import defaultdict, deque
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
from statistics import median

DEFAULT_RULES = Path(__file__).resolve().parents[2] / "data" / "samples" / "monitor_rules.json"


def instant(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp precisa incluir fuso horário")
    return parsed.astimezone(timezone.utc)


def number(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Leitura numérica inválida")
    return parsed


def boolean(value):
    if value in (True, "true", "True", "1", 1):
        return True
    if value in (False, "false", "False", "0", 0):
        return False
    raise ValueError("Booleano inválido")


def load_events(path, kind=None):
    """JSONL suporta fluxo misto; CSV usa kind informado e o esquema documentado."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            if kind is None:
                raise ValueError("CSV exige a modalidade")
            for row in csv.DictReader(stream):
                if kind == "vitals":
                    metrics = {k: v for k, v in row.items() if k not in {"timestamp", "patient_id"} and v != ""}
                    yield {"kind": kind, "timestamp": row["timestamp"], "patient_id": row["patient_id"], "metrics": metrics}
                else:
                    yield {**row, "kind": kind}
        else:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSON inválido na linha {line_no}") from exc


class AnomalyEngine:
    def __init__(self, rules=None):
        self.rules = rules or json.loads(DEFAULT_RULES.read_text(encoding="utf-8"))
        self.history = defaultdict(lambda: deque(maxlen=self.rules["window"]))
        self.last = {}
        self.prescriptions = {}
        self.patient_prescriptions = set()
        self.low_since = {}
        self.quality = []

    def process(self, event):
        pid = str(event.get("patient_id", "")).strip()
        kind = event.get("kind")
        if not pid or kind not in {"vitals", "prescription", "movement"}:
            raise ValueError("patient_id e kind válidos são obrigatórios")
        stamp = instant(event["timestamp"])
        key = (pid, kind, str(event.get("medication", "")).strip().lower())
        previous = self.last.get(key)
        if previous is not None and stamp <= previous:
            raise ValueError("Eventos fora de ordem ou duplicados para o mesmo paciente/modalidade")
        alerts = []

        def alert(alert_type, severity, message, evidence, discriminator):
            alerts.append({"type": alert_type, "severity": severity, "message": message,
                           "patient_id": pid, "source": kind, "timestamp": stamp.isoformat(),
                           "evidence": evidence, "score": 1. if severity == "CRITICO" else .5,
                           "episode_key": f"{pid}:{kind}:{discriminator}"})

        if kind == "vitals":
            metrics = event.get("metrics") or {}
            if not metrics:
                raise ValueError("Registro sem sinais vitais")
            parsed = {metric: number(value) for metric, value in metrics.items()}
            if set(parsed) - self.rules["vitals"].keys():
                raise ValueError("Sinal vital/unidade não configurado nas regras")
            missing = sorted(self.rules["vitals"].keys() - set(parsed))
            if missing:
                self.quality.append({"patient_id": pid, "timestamp": stamp.isoformat(),
                                     "reason": "missing_vital_metrics", "metrics": missing})
            for metric, value in parsed.items():
                rule = self.rules["vitals"][metric]
                history = self.history[(pid, metric)]
                outside = value < rule["min"] or value > rule["max"]
                baseline = median(history) if history else None
                z = None
                if len(history) >= self.rules["min_history"]:
                    mad = median(abs(v - baseline) for v in history)
                    scale = max(1.4826 * mad, rule["min_scale"])
                    z = abs(value - baseline) / scale
                if outside or (z is not None and z >= self.rules["robust_z"]):
                    alert("anomalia_sinal_vital", rule.get("severity", "ATENCAO") if outside else "ATENCAO",
                          f"Desvio em {metric} conforme configuração da demonstração.",
                          {"metric": metric, "value": value, "expected_min": rule["min"],
                           "expected_max": rule["max"], "baseline_median": baseline,
                           "robust_z": round(z, 3) if z is not None else None,
                           "method": "configured_limits" if outside else "rolling_median_MAD"}, metric)
                else:
                    history.append(value)
        elif kind == "prescription":
            medication = str(event.get("medication", "")).strip().lower()
            if not medication:
                raise ValueError("Prescrição exige medicamento")
            planned = boolean(event.get("planned_change", False))
            if planned and not str(event.get("reason", "")).strip():
                raise ValueError("Mudança planejada exige justificativa")
            action = event.get("action", "set")
            if action not in {"set", "stop"}:
                raise ValueError("Ação de prescrição inválida")
            rx_key = (pid, medication)
            old = self.prescriptions.get(rx_key)
            if action == "stop":
                if old and not planned:
                    alert("alteracao_prescricao", "ATENCAO", "Retirada sem justificativa registrada.",
                          {"medication": medication, "previous": old, "action": "stop"}, medication)
                self.prescriptions.pop(rx_key, None)
            else:
                unit = event.get("unit", "").lower()
                factors = {"mg": 1., "g": 1000., "mcg": .001}
                if unit not in factors:
                    raise ValueError("Unidade de dose não suportada: use mg, g ou mcg")
                dose, frequency = number(event["dose"]), number(event["frequency_per_day"])
                if dose <= 0 or frequency <= 0:
                    raise ValueError("Dose e frequência devem ser positivas")
                current = {"dose_mg": dose * factors[unit], "frequency_per_day": frequency}
                ratio = max(current["dose_mg"] / old["dose_mg"], old["dose_mg"] / current["dose_mg"],
                            frequency / old["frequency_per_day"], old["frequency_per_day"] / frequency) if old else None
                new_unexpected = old is None and pid in self.patient_prescriptions
                if not planned and (new_unexpected or (ratio is not None and ratio >= self.rules["prescription_ratio"])):
                    alert("alteracao_prescricao", "ATENCAO", "Mudança inesperada exige revisão da equipe.",
                          {"medication": medication, "previous": old, "current": current,
                           "change_ratio": ratio, "reason": "new_medication" if new_unexpected else "dose_or_frequency"}, medication)
                self.prescriptions[rx_key] = current
                self.patient_prescriptions.add(pid)
        else:
            activity = number(event["activity_score"])
            if not 0 <= activity <= 1:
                raise ValueError("activity_score deve estar entre 0 e 1")
            present = boolean(event["patient_present"])
            quality = boolean(event.get("quality_good", True))
            if previous and (stamp - previous).total_seconds() > self.rules["movement_max_gap_seconds"]:
                self.low_since.pop(pid, None)
            if not present or not quality:
                self.low_since.pop(pid, None)
                self.quality.append({"patient_id": pid, "timestamp": stamp.isoformat(),
                                     "reason": "movement_unavailable"})
            elif activity <= self.rules["movement_low_threshold"]:
                since = self.low_since.setdefault(pid, stamp)
                seconds = (stamp - since).total_seconds()
                if seconds >= self.rules["immobility_seconds"]:
                    alert("imobilidade_prolongada", "ATENCAO", "Baixa atividade persistente com paciente presente.",
                          {"duration_seconds": seconds, "activity_score": activity,
                           "threshold_seconds": self.rules["immobility_seconds"],
                           "method": "temporal_low_activity"}, "immobility")
            else:
                self.low_since.pop(pid, None)
        self.last[key] = stamp
        return alerts
