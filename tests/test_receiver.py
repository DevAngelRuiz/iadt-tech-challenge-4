import json
from threading import Thread
from urllib.request import Request, urlopen
from src.alert_server import create_server
from src.reporting.alert_store import AlertStore
from src.monitor import run_events


def test_alert_visible_before_end_of_stream_and_acknowledged(tmp_path):
    store = AlertStore(tmp_path / "alerts.sqlite")
    server = create_server(store, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"

    def events():
        yield {"kind": "vitals", "patient_id": "demo", "timestamp": "2026-01-01T00:00:00Z", "metrics": {"spo2_pct": 88}}
        # O produtor ainda não terminou, mas o receptor já precisa disponibilizar o alerta.
        with urlopen(root + "/api/alerts", timeout=5) as response:
            alerts = json.load(response)
        assert len(alerts) == 1
        request = Request(root + "/api/alerts/" + alerts[0]["id"] + "/ack", method="POST")
        with urlopen(request, timeout=5) as response:
            assert response.status == 200
        assert store.list()[0]["acknowledged_at"]
        yield {"kind": "vitals", "patient_id": "demo", "timestamp": "2026-01-01T00:00:10Z", "metrics": {"spo2_pct": 98}}

    try:
        assert run_events(events(), store)["records"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
