"""Análise contextual via Qwen em servidor llama.cpp local; sem fallback oculto."""
from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from src.text.qwen_prompt import INSTRUCTIONS, EXAMPLES

CATEGORIES = ["fadiga", "respiratorio", "dificuldade_fala", "sofrimento_emocional", "ansiedade", "violencia"]
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "findings": {"type": "array", "maxItems": 12, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES},
                "assertion": {"type": "string", "enum": ["present", "negated", "uncertain"]},
                "evidence": {"type": "string"},
            }, "required": ["category", "assertion", "evidence"]}},
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED", "UNKNOWN"]},
    }, "required": ["findings", "summary", "sentiment"],
}
PROMPT_VERSION = "clinical-extraction-v3-grounded"
SYSTEM_PROMPT = INSTRUCTIONS + json.dumps(SCHEMA, ensure_ascii=False)


def _schema_for_text(text: str) -> dict:
    # Apenas segmentação do texto, sem inferir categorias ou afirmações.
    # A enumeração impede paráfrases e mudanças de maiúsculas nas citações.
    excerpts = [text]
    fragments = re.split(
        r"[,;.!?\n]+\s*(?:(?:e|mas|porém|contudo|entretanto)\s+)?|\s+(?:e|mas|porém|contudo|entretanto)\s+",
        text, flags=re.IGNORECASE)
    for fragment in fragments:
        fragment = fragment.strip()
        if fragment and fragment not in excerpts:
            excerpts.append(fragment)
        if len(excerpts) >= 65:
            break
    schema = deepcopy(SCHEMA)
    schema["properties"]["findings"]["items"]["properties"]["evidence"]["enum"] = excerpts
    return schema


def _messages(text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example, answer in EXAMPLES:
        messages.extend([{"role": "user", "content": example},
                         {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}])
    messages.append({"role": "user", "content": text})
    return messages


def _validate(data: dict, text: str) -> dict:
    if not isinstance(data, dict) or set(data) != {"summary", "sentiment", "findings"}:
        raise ValueError("invalid_schema")
    if not isinstance(data["summary"], str) or len(data["summary"]) > 2000:
        raise ValueError("invalid_summary")
    if data["sentiment"] not in SCHEMA["properties"]["sentiment"]["enum"]:
        raise ValueError("invalid_sentiment")
    if not isinstance(data["findings"], list) or len(data["findings"]) > 12:
        raise ValueError("invalid_findings")
    for item in data["findings"]:
        if not isinstance(item, dict) or set(item) != {"category", "assertion", "evidence"}:
            raise ValueError("invalid_finding")
        if item["category"] not in CATEGORIES or item["assertion"] not in {"present", "negated", "uncertain"}:
            raise ValueError("invalid_category_or_assertion")
        if not isinstance(item["evidence"], str) or not item["evidence"].strip() or item["evidence"] not in text:
            raise ValueError("ungrounded_evidence")
    return data


def analyze_qwen_text(text: str) -> dict:
    model = os.getenv("QWEN_MODEL", "qwen2.5-1.5b-instruct")
    result = {"engine": "qwen-local", "model": model, "status": "unavailable",
              "prompt_version": PROMPT_VERSION,
              "cloud_executed": False, "model_inference_executed": False,
              "sentiment": None, "sentiment_score": {}, "entities": [], "key_phrases": [],
              "findings": [], "summary": "", "evidence_validated": False,
              "limitations": ["Saída de LLM para revisão; evidência literal não garante interpretação correta.",
                              "Resumo e sentimento são gerados pelo modelo; não são diagnóstico."]}
    text = (text or "").strip()
    if not text:
        return dict(result, status="empty")
    if len(text) > 6000:
        return dict(result, status="input_too_long")
    base = os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8081/v1").rstrip("/")
    try:
        url = urlsplit(base)
    except ValueError:
        return dict(result, status="invalid_endpoint")
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"} or url.username or url.password or url.query or url.fragment:
        return dict(result, status="invalid_endpoint")
    request = Request(base + "/chat/completions", data=json.dumps({
        "model": model, "messages": _messages(text),
        "temperature": 0, "seed": 42, "max_tokens": 768, "stream": False,
        "response_format": {"type": "json_schema", "json_schema": {"name": "clinical_observations", "strict": True, "schema": _schema_for_text(text)}},
    }).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=180) as response:
            raw = response.read(131073)
        if len(raw) > 131072:
            raise ValueError("response_too_large")
        payload = json.loads(raw)
        choice = payload["choices"][0]
        result["model_inference_executed"] = True
        result["served_model"] = payload.get("model")
        if payload.get("model") != model:
            return dict(result, status="model_mismatch")
        if choice.get("finish_reason") != "stop":
            return dict(result, status="incomplete_response")
        data = _validate(json.loads(choice["message"]["content"]), text)
        return dict(result, **data, key_phrases=list(dict.fromkeys(f["evidence"] for f in data["findings"])),
                    status="ok", evidence_validated=True)
    except (URLError, TimeoutError, OSError):
        return dict(result, status="unavailable", error="local_server_unavailable")
    except ValueError as exc:
        # Somente códigos internos são expostos; nunca copiar o conteúdo bruto
        # de uma resposta inválida ou de uma exceção de parser para o relatório.
        reason = str(exc)
        known = {"invalid_schema", "invalid_summary", "invalid_sentiment", "invalid_findings",
                 "invalid_finding", "invalid_category_or_assertion", "ungrounded_evidence", "response_too_large"}
        return dict(result, status="invalid_response", error="schema_or_evidence_validation_failed",
                    validation_error=reason if reason in known else "invalid_json_or_response")
    except (KeyError, IndexError, TypeError):
        return dict(result, status="invalid_response", error="schema_or_evidence_validation_failed")
