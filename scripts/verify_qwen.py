"""Verificação real de extração textual; não usa respostas simuladas."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_environment
from src.text.qwen_nlp import analyze_qwen_text

CASES = [
    ("presente", "Estou cansada e tenho falta de ar.", {("fadiga", "present"), ("respiratorio", "present")}),
    ("negado", "Não tenho falta de ar e não estou cansada.", {("respiratorio", "negated"), ("fadiga", "negated")}),
    ("contraste", "Estou cansada, mas não tenho falta de ar.", {("fadiga", "present"), ("respiratorio", "negated")}),
    ("normal", "Consulta de rotina. Estou bem.", set()),
    ("incerto", "Talvez eu esteja com falta de ar, não tenho certeza.", {("respiratorio", "uncertain")}),
]


def main():
    load_environment()
    results = []
    for name, text, expected in CASES:
        data = analyze_qwen_text(text)
        observed = {(f["category"], f["assertion"]) for f in data["findings"]}
        passed = data["status"] == "ok" and observed == expected
        results.append({"case": name, "text": text, "expected": sorted(expected),
                        "passed": passed, "result": data})
        print(f"{name}: {'PASSOU' if passed else 'FALHOU'} / {data['status']}", flush=True)
        if not passed:
            print("  esperado: " + json.dumps(sorted(expected), ensure_ascii=False), flush=True)
            print("  recebido: " + json.dumps(sorted(observed), ensure_ascii=False), flush=True)
            if data.get("validation_error"):
                print("  validação: " + data["validation_error"], flush=True)
        if data["status"] == "unavailable":
            break
    report = {"all_passed": len(results) == len(CASES) and all(r["passed"] for r in results),
              "scope": "Cinco exemplos sintéticos; não é validação clínica nem benchmark geral.", "cases": results}
    out = ROOT / "outputs" / "qwen_validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["all_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
