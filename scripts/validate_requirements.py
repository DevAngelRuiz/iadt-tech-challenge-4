"""Gera evidências locais por CLI e verifica entrega progressiva no receptor HTTP."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Thread
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.alert_server import create_server
from src.reporting.alert_store import AlertStore
from src.cloud.azure_services import azure_readiness

OUT = ROOT / "outputs" / "requirements_validation"
EXAMPLES = ROOT / "outputs" / "examples" / "requisitos"


def run(name, arguments, env):
    process = subprocess.run([sys.executable, "-X", "utf8", *arguments], cwd=ROOT, env=env,
                             text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=300)
    (OUT / f"{name}.log").write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"{name} terminou com código {process.returncode}; consulte o log")
    return process


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OPENAI_API_KEY="", AWS_ACCESS_KEY_ID="", AWS_SECRET_ACCESS_KEY="",
               AZURE_SPEECH_KEY="", AZURE_LANGUAGE_KEY="", REPORT_PROVIDER="local", PYTHONUTF8="1",
               YOLO_CONFIG_DIR=str(OUT / "ultralytics_config"))
    evidence = {"azure_live_validation": "pending_resources", "configured": azure_readiness(), "cases": {}}
    for name in ["normal", "anomalo"]:
        print(f"Executando monitor {name}", flush=True)
        run("monitor_" + name, ["-m", "src.monitor", "--events", f"data/samples/monitoring/{name}.jsonl",
                               "--interval", "0", "--store", str(OUT / (name + ".sqlite")),
                               "--output", str(OUT / ("monitor_" + name + ".json"))], env)
        payload = json.loads((OUT / ("monitor_" + name + ".json")).read_text(encoding="utf-8"))
        evidence["cases"][name] = {"records": payload["records"], "alerts": len(payload["alerts"]),
                                  "types": sorted({a["type"] for a in payload["alerts"]}), "suppressed": payload["suppressed"]}
        shutil.copy2(OUT / ("monitor_" + name + ".json"), EXAMPLES)
    assert evidence["cases"]["normal"]["alerts"] == 0
    assert set(evidence["cases"]["anomalo"]["types"]) == {"anomalia_sinal_vital", "alteracao_prescricao", "imobilidade_prolongada"}

    store_path = OUT / "receiver.sqlite"
    store = AlertStore(store_path)
    initial_ids = {alert["id"] for alert in store.list()}
    server = create_server(store, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    started = time.perf_counter()
    seen_before_end = False
    with (OUT / "live_receiver.log").open("w", encoding="utf-8") as log:
        producer = subprocess.Popen([sys.executable, "-m", "src.monitor", "--events", "data/samples/monitoring/anomalo.jsonl",
                                     "--interval", ".1", "--store", str(store_path), "--output", str(OUT / "live_monitor.json")],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while producer.poll() is None and time.perf_counter() - started < 90:
                with urlopen(root + "/api/alerts", timeout=5) as response:
                    alerts = [alert for alert in json.load(response) if alert["id"] not in initial_ids]
                if alerts and producer.poll() is None:
                    seen_before_end = True
                    with urlopen(Request(root + "/api/alerts/" + alerts[0]["id"] + "/ack", method="POST"), timeout=5):
                        pass
                time.sleep(.1)
            if producer.poll() is None:
                producer.kill()
            assert producer.wait(timeout=10) == 0
            assert seen_before_end
            assert any(a["acknowledged_at"] for a in store.list())
            evidence["receiver"] = {"visible_before_producer_finished": True, "acknowledged": True}
        finally:
            if producer.poll() is None:
                producer.kill()
                producer.wait()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    print("Executando pipeline integrado com YOLO e as três modalidades estruturadas", flush=True)
    run("integrado", ["-m", "src.main", "--video", "data/samples/consulta_com_sangramento.mp4",
         "--text-file", "data/samples/consulta_pos_parto.txt", "--transcript-override", "Estou triste e chorando, sem energia",
         "--speech-provider", "offline", "--nlp-provider", "local", "--skip-emotion", "--skip-pose",
         "--vitals", "data/samples/monitoring/sinais_vitais_anomalo.csv",
         "--prescriptions", "data/samples/monitoring/prescricoes_anomalo.csv",
         "--movement", "data/samples/monitoring/movimentacao_anomalo.csv",
         "--alert-db", str(OUT / "integrado.sqlite"), "--output-dir", str(OUT / "integrado"), "--prefix", "integrado"], env)
    path = sorted((OUT / "integrado").glob("integrado_*.json"))[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["video"]["yolo"]["model"] == "YOLOv8-bleeding"
    assert payload["video"]["yolo"]["coverage"] == 1
    assert payload["fusion"]["alert_level"] == "CRITICO"
    evidence["integrated"] = {"json": str(path.relative_to(ROOT)), "model": payload["video"]["yolo"]["model"],
                              "coverage": 1, "level": payload["fusion"]["alert_level"],
                              "types": sorted({a["type"] for a in payload["fusion"]["alerts"]})}
    shutil.copy2(path, EXAMPLES / "integrado.json")
    (EXAMPLES / "integrado.md").write_text(path.with_suffix(".md").read_text(encoding="utf-8").replace(path.name, "integrado.json"), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXAMPLES / "summary.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
