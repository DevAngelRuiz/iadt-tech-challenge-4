import pytest


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    # Testes unitários nunca acionam contas reais que estejam configuradas no notebook.
    for key in ["OPENAI_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_LANGUAGE_KEY", "AZURE_LANGUAGE_ENDPOINT"]:
        # Valores presentes e vazios impedem que load_dotenv(override=False)
        # reintroduza credenciais reais quando um verificador recarrega o .env.
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("NLP_PROVIDER", "local")
    monkeypatch.setenv("SPEECH_PROVIDER", "offline")
    monkeypatch.setenv("REPORT_PROVIDER", "local")
    monkeypatch.setenv("QWEN_BASE_URL", "http://127.0.0.1:8081/v1")
    monkeypatch.setenv("QWEN_MODEL", "qwen2.5-1.5b-instruct")
