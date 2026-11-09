import json
from unittest.mock import Mock
from urllib.error import URLError
import pytest
from src.text import qwen_nlp as qwen


TEXT = "Estou cansada, mas não tenho falta de ar."


def answer(findings=None):
    return {"summary": "Relata cansaço e nega falta de ar.", "sentiment": "NEUTRAL",
            "findings": findings if findings is not None else [
                {"category": "fadiga", "assertion": "present", "evidence": "Estou cansada"},
                {"category": "respiratorio", "assertion": "negated", "evidence": "não tenho falta de ar"}]}


def mock_server(monkeypatch, data=None, finish="stop", model="qwen2.5-1.5b-instruct"):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps({"model": model, "choices": [
        {"finish_reason": finish, "message": {"content": json.dumps(data if data is not None else answer())}}]}).encode()
    opener = Mock(return_value=response)
    monkeypatch.setattr(qwen, "urlopen", opener)
    return opener


def test_real_request_contract_and_grounded_negations(monkeypatch):
    opener = mock_server(monkeypatch)
    result = qwen.analyze_qwen_text(TEXT)
    assert result["status"] == "ok" and result["model_inference_executed"]
    assert result["evidence_validated"] and not result["cloud_executed"]
    assert result["findings"][1]["assertion"] == "negated"
    request = opener.call_args.args[0]
    body = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:8081/v1/chat/completions"
    request_schema = body["response_format"]["json_schema"]["schema"]
    excerpts = request_schema["properties"]["findings"]["items"]["properties"]["evidence"].pop("enum")
    assert request_schema == qwen.SCHEMA
    assert set(excerpts) == {TEXT, "Estou cansada", "não tenho falta de ar"}
    # O llama.cpp restringe a geração pelo esquema, mas não o inclui no
    # contexto do modelo. O contrato deve estar explícito no prompt também.
    assert json.dumps(qwen.SCHEMA, ensure_ascii=False) in body["messages"][0]["content"]
    assert result["prompt_version"] == qwen.PROMPT_VERSION
    assert body["messages"][-1]["content"] == TEXT
    for index, message in enumerate(body["messages"]):
        if message["role"] == "assistant":
            qwen._validate(json.loads(message["content"]), body["messages"][index - 1]["content"])


@pytest.mark.parametrize("data", [
    {"summary": "inventado"},
    answer([{"category": "fadiga", "assertion": "present", "evidence": "está exausta"}]),
    answer([{"category": "diagnostico", "assertion": "present", "evidence": "Estou cansada"}]),
    answer([{"category": "fadiga", "assertion": "present", "evidence": ""}]),
])
def test_invalid_output_cannot_be_presented_as_success(monkeypatch, data):
    mock_server(monkeypatch, data)
    result = qwen.analyze_qwen_text(TEXT)
    assert result["status"] == "invalid_response" and not result["findings"]
    assert not result["evidence_validated"]


@pytest.mark.parametrize("finish,model,status", [("length", "qwen2.5-1.5b-instruct", "incomplete_response"),
                                                ("stop", "other-model", "model_mismatch")])
def test_incomplete_or_wrong_model_rejected(monkeypatch, finish, model, status):
    mock_server(monkeypatch, finish=finish, model=model)
    assert qwen.analyze_qwen_text(TEXT)["status"] == status


def test_unavailable_has_no_hidden_fallback(monkeypatch):
    monkeypatch.setattr(qwen, "urlopen", Mock(side_effect=URLError("local unavailable")))
    out = qwen.analyze_qwen_text(TEXT)
    assert out["status"] == "unavailable" and out["engine"] == "qwen-local"
    assert not out["model_inference_executed"]


def test_invalid_json_diagnostics_do_not_echo_model_output(monkeypatch):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps({"model": "qwen2.5-1.5b-instruct", "choices": [
        {"finish_reason": "stop", "message": {"content": "private-example-not-json"}}]}).encode()
    monkeypatch.setattr(qwen, "urlopen", Mock(return_value=response))
    result = qwen.analyze_qwen_text(TEXT)
    assert result["status"] == "invalid_response"
    assert result["validation_error"] == "invalid_json_or_response"
    assert "private-example" not in json.dumps(result)


def test_excerpt_constraints_are_literal_and_do_not_leak_between_requests():
    first = qwen._schema_for_text(TEXT)
    second_text = "Relata fala arrastada; não há tristeza."
    second = qwen._schema_for_text(second_text)
    leaf = lambda schema: schema["properties"]["findings"]["items"]["properties"]["evidence"]
    assert "enum" not in leaf(qwen.SCHEMA)
    assert all(excerpt in TEXT and excerpt for excerpt in leaf(first)["enum"])
    assert all(excerpt in second_text and excerpt for excerpt in leaf(second)["enum"])
    assert TEXT not in leaf(second)["enum"]


@pytest.mark.parametrize("base", ["https://external.example/v1", "http://[bad", "http://user:pass@localhost/v1"])
def test_only_local_endpoint(monkeypatch, base):
    monkeypatch.setenv("QWEN_BASE_URL", base)
    opener = Mock()
    monkeypatch.setattr(qwen, "urlopen", opener)
    assert qwen.analyze_qwen_text(TEXT)["status"] == "invalid_endpoint"
    opener.assert_not_called()


def test_no_silent_truncation_or_empty_inference(monkeypatch):
    opener = Mock()
    monkeypatch.setattr(qwen, "urlopen", opener)
    assert qwen.analyze_qwen_text("")["status"] == "empty"
    assert qwen.analyze_qwen_text("a" * 6001)["status"] == "input_too_long"
    opener.assert_not_called()


def test_fusion_uses_llm_evidence_separately_from_rules(monkeypatch):
    from src.fusion.multimodal_engine import fuse_modalities
    mock_server(monkeypatch)
    result = fuse_modalities(text_input=TEXT, nlp_provider="qwen")
    alerts = [a for a in result["alerts"] if a["type"] == "revisao_textual_llm"]
    assert len(alerts) == 1
    assert [e["category"] for e in alerts[0]["evidence"]] == ["fadiga"]
    assert result["text_classification"]["model"] == "lexical-rules-with-negation"
    assert "Resumo gerado" in result["report"]["report"]
    mock_server(monkeypatch, answer([{"category": "respiratorio", "assertion": "negated", "evidence": "não tenho falta de ar"}]))
    negative = fuse_modalities(text_input="não tenho falta de ar", nlp_provider="qwen")
    assert negative["alerts"] == []


def test_setup_blocks_download_when_disk_is_low(monkeypatch, tmp_path):
    from scripts import setup_qwen
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(setup_qwen.shutil, "disk_usage", lambda _: usage(10, 9, 1))
    download = Mock()
    monkeypatch.setattr(setup_qwen, "download", download)
    assert setup_qwen.main(["--directory", str(tmp_path / "qwen"), "--download"]) == 4
    download.assert_not_called()


def test_strict_cli_requires_qwen_provider(tmp_path):
    from src.main import main
    assert main(["--text", TEXT, "--nlp-provider", "local", "--require-qwen", "--output-dir", str(tmp_path)]) == 3
