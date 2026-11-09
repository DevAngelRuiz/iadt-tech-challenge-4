# Whisper local e adaptação indicada pelo Prof. Jose Neto

Na comunidade do curso, o **Prof. Jose Neto indicou Whisper como alternativa ao Azure Speech**, e Llama 3.1/3.2 ou Qwen 2.5 como alternativas ao Azure AI Language, conforme informação trazida pela equipe do projeto. Essa orientação é registrada como motivação da adaptação, sem atribuir ao professor uma frase literal.

Implementamos **Whisper local**, com inferência comprovada. O provedor Azure foi preservado como alternativa opcional. Qwen também foi instalado e executado: passou nos cinco controles originais e em seis de oito frases adicionais. Consulte [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md) para os resultados e erros observados e [GUIA_QWEN.md](GUIA_QWEN.md) para repetir a execução. O provedor textual padrão continua sendo regras locais; a flag `--nlp-provider qwen` seleciona o modelo explicitamente.

## Componentes

- `src/audio/whisper_transcriber.py`: carrega o modelo e transcreve o áudio localmente, em português.
- `src/audio/transcriber.py`: seleciona `whisper`, `azure`, `offline` ou `legacy` explicitamente.
- `src/main.py`: Whisper é o padrão; `--require-transcription` exige fala transcrita automaticamente e rejeita override manual.
- `src/audio/voice_risk.py`: registra modelo, dispositivo, segmentos, execução real e resultado da transcrição.
- `.env.example`: configura modelo, CPU e pasta de cache; não é necessária chave para Whisper local.

YOLO, detectores temporais, fusão e painel continuam sendo utilizados. A análise textual local não é um LLM; essa substituição é uma etapa separada.

## Configuração

```dotenv
SPEECH_PROVIDER=whisper
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
WHISPER_CACHE_DIR=models/whisper
NLP_PROVIDER=local
REPORT_PROVIDER=local
```

O modelo `base` é multilíngue e foi escolhido como ponto de partida leve para este notebook. A primeira execução pode baixar os pesos; depois eles ficam no cache local. Não é necessário instalar Ollama para o Whisper. Os pesos ficam fora do versionamento Git.

Instalação e verificação:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest
.\.venv\Scripts\python.exe scripts/verify_whisper.py
```

O verificador sem áudio só mostra a configuração; não comprova transcrição.

## Preparar fala fictícia para a demonstração

Se houver voz pt-BR no Windows, o script abaixo cria um WAV sintético e seu texto de referência. Não usa pacientes reais nem um serviço pago:

```powershell
.\scripts\prepare_spoken_sample.ps1
```

Também é possível gravar uma fala autorizada. O controle `consulta_hesitante.wav` contém tons e pausas e não deve ser apresentado como fala de uma consulta.

## Executar sem transcrição manual

```powershell
.\.venv\Scripts\python.exe scripts/verify_whisper.py --audio data/samples/consulta_falada_sintetica.wav
```

Conferir no JSON gerado em `outputs/whisper_validation/`:

- `audio.stt_engine`: `whisper-local`;
- `audio.status`: `ok`;
- `audio.transcript`: texto não vazio;
- `audio.model_inference_executed`: `true`;
- `audio.cloud_executed`: `false`, pois a execução é local;
- `audio.stt_model` e `audio.stt_device`: modelo e dispositivo usados.

Código de saída 3 significa que a transcrição automática não foi comprovada. A ausência de fala e falhas de pacote, modelo ou áudio são estados distintos. Segmentos com baixa qualidade são filtrados; isso não elimina a possibilidade de erro ou alucinação e não constitui validação clínica.

Os WAVs são normalizados para PCM mono/16 kHz e entregues como vetor ao Whisper. Assim, o Whisper não precisa localizar FFmpeg no PATH para essa etapa. MoviePy continua responsável pela extração de outros formatos suportados.

## Referência

A referência oficial do modelo e da biblioteca é o [repositório OpenAI Whisper](https://github.com/openai/whisper). A transcrição é executada localmente, sem uso de serviços gerenciados em nuvem.
