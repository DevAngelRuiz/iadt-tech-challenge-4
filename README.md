# Tech Challenge Fase 4 — Monitoramento multimodal

Protótipo acadêmico de saúde da mulher com vídeo, áudio, texto e monitoramento temporal. O enunciado exige os três blocos técnicos; o estado de cada integração é registrado nas saídas.

**Adaptação indicada pelo Prof. Jose Neto na comunidade do curso:** Whisper para substituir Azure Speech; Llama 3.1/3.2 ou Qwen 2.5 como alternativas ao Azure AI Language, conforme orientação informada pela equipe. O projeto oferece Whisper e Qwen locais, além dos provedores Azure opcionais. Consulte os guias de instalação e execução do [Whisper](GUIA_WHISPER.md), do [Qwen](GUIA_QWEN.md) e do [Azure](GUIA_AZURE.md). As limitações estão no [relatório técnico](RELATORIO_TECNICO.md).

## Estado da entrega

- Vídeo: YOLOv8 customizado para regiões de sangramento, vídeo anotado e relatório. Emoção dispõe de heurística identificada; postura exige MediaPipe com articulações visíveis. O fallback OpenCV mede atividade global e registra postura indisponível, sem gerar alerta de postura.
- Áudio: extração PCM, Whisper local e indicadores vocais condicionados a fala reconhecida; Azure é um provedor alternativo. Áudio ausente não é substituído por silêncio.
- Texto: regras locais com negação e Qwen local com resumo/achados evidenciados, executado nos controles sintéticos. Azure Text Analytics preservado como alternativa. Scores de regras e achados de LLM são identificados separadamente; o modelo pequeno apresentou erros nas frases adicionais.
- Anomalias: sinais vitais (limites e mediana/MAD), mudanças de prescrições e baixa atividade persistente.
- Alertas: receptor SQLite, deduplicação, painel local e confirmação de leitura. O monitor publica durante o fluxo.
- Apresentação: o link do vídeo de até 15 minutos será informado no PDF de entrega.

Instalação do SDK e testes simulados não comprovam integração real com Azure. O JSON informa `engine`, `status` e `cloud_executed`.

## Instalação no Windows

Após clonar o repositório, abra o PowerShell na pasta que contém este `README.md` e `requirements.txt`. Todos os comandos abaixo partem dessa pasta, inclusive os executados em um segundo terminal.

O projeto foi validado com **Python 3.12.3**. Use essa versão com `venv` e `pip` para reproduzir o ambiente validado. É necessário acesso à internet para instalar pacotes e baixar os modelos na primeira execução. O provisionamento Qwen deste projeto usa o pacote **Windows x64 para CPU**. Docker e Ollama não são necessários para o caminho local Whisper/Qwen.

Crie o ambiente e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest
```

Os comandos usam diretamente o Python da `.venv`; não é necessário ativar o ambiente. Prepare a configuração sem substituir um `.env` existente e gere os exemplos de monitoramento:

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
.\.venv\Scripts\python.exe scripts/prepare_monitoring_data.py
```

O `.env.example` usa `SPEECH_PROVIDER=whisper`, `NLP_PROVIDER=local` e `REPORT_PROVIDER=local`. Assim, o primeiro teste pode usar Whisper e regras textuais sem credenciais. Os comandos com flags explícitas permitem escolher outro provedor sem alterar esse padrão. Não versione o `.env` nem suas chaves.

Instalar os pacotes Python e obter os pesos dos modelos são etapas diferentes:

| Componente | Como preparar |
|---|---|
| YOLO | O projeto inclui `models/bleeding_yolov8n.pt` e exemplos de vídeo; não é necessário retreinar |
| Whisper | O pacote vem de `requirements.txt`; o modelo `base` é baixado na primeira transcrição e reutilizado de `models/whisper/` |
| Qwen + executor llama.cpp | Baixados pelo script `setup_qwen.py`, conforme a seção abaixo; ficam em `models/qwen/` |
| Azure Speech / Language | Os SDKs vêm de `requirements.txt`; recursos e credenciais são opcionais e seguem o [guia Azure](GUIA_AZURE.md) |

Os downloads do Whisper e do Qwen ficam fora do Git. A instalação de pacotes também ocupa espaço: confira o disco antes de começar. O provisionamento Qwen estima cerca de 1,35 GB adicionais para modelo/programa e mantém 2 GiB de reserva; a estimativa não inclui o ambiente Python nem o modelo Whisper.

## Demonstração do monitoramento

Em um terminal, inicie o receptor:

```powershell
.\.venv\Scripts\python.exe -m src.alert_server
```

Abra http://127.0.0.1:8765 e mantenha o painel visível. Em outro terminal:

```powershell
.\.venv\Scripts\python.exe -m src.monitor --events data/samples/monitoring/normal.jsonl --output outputs/monitor_normal.json
.\.venv\Scripts\python.exe -m src.monitor --events data/samples/monitoring/anomalo.jsonl --output outputs/monitor_anomalo.json --interval 0.2
```

O normal deve produzir zero alertas. O anômalo deve mostrar sinais vitais, alteração de prescrição e imobilidade no painel, com evidências e confirmação de leitura. O banco padrão é `outputs/alerts.sqlite`; execuções têm identificadores distintos. `--follow` acompanha novas linhas completas de um JSONL até Ctrl+C.

Essa é uma reprodução de telemetria sintética, processada progressivamente. O campo de latência mede processamento até enfileiramento local; não mede atraso de sensores ou resposta clínica. O painel consulta o receptor a cada segundo.

## Pipeline multimodal e dados estruturados

```powershell
.\.venv\Scripts\python.exe -m src.main --video data/samples/consulta_com_sangramento.mp4 --text-file data/samples/consulta_pos_parto.txt --transcript-override "Estou triste e chorando, não consigo dormir, sem energia" --speech-provider offline --nlp-provider local --vitals data/samples/monitoring/sinais_vitais_anomalo.csv --prescriptions data/samples/monitoring/prescricoes_anomalo.csv --movement data/samples/monitoring/movimentacao_anomalo.csv --prefix integrado
```

Esse comando é uma demonstração local com transcrição manual: não comprova STT. Gera JSON, Markdown, vídeo anotado e alertas no receptor. Os vídeos são analisados integralmente por padrão; `--max-video-frames` permite um limite explícito, registrado na cobertura.

Esquemas e significado dos dados: [data/samples/monitoring/README.md](data/samples/monitoring/README.md). Limiares são configuráveis em `data/samples/monitor_rules.json` e servem apenas ao cenário acadêmico.

## Comprovar a transcrição Whisper

O projeto inclui `data/samples/consulta_falada_sintetica.wav`, uma fala fictícia em português. Use esse arquivo para o primeiro teste:

```powershell
.\.venv\Scripts\python.exe scripts/verify_whisper.py --audio data/samples/consulta_falada_sintetica.wav
```

A primeira execução pode baixar os pesos do modelo `base`; as seguintes usam o cache. Conferir `audio.stt_engine=whisper-local`, `status=ok`, transcrição não vazia e `model_inference_executed=true`. `cloud_executed=false` identifica a execução local. Código de saída 3 significa que a transcrição automática exigida não foi comprovada. `consulta_hesitante.wav` contém tons, não fala.

JSON e Markdown são gerados em `outputs/whisper_validation/`. Para recriar a fala sintética, opcionalmente execute `.\scripts\prepare_spoken_sample.ps1`; essa preparação exige uma voz pt-BR instalada no Windows. Detalhes: [GUIA_WHISPER.md](GUIA_WHISPER.md).

## Preparar e executar Qwen local

O script `scripts/verify_qwen.py` executa cinco controles sintéticos com o modelo real. Quem clonar o projeto precisa instalar os arquivos locais e executar a verificação abaixo; os pesos não estão no Git. Testes automatizados com respostas simuladas não substituem a execução real.

**1. Conferir o espaço e baixar o modelo e o executor.** Na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe scripts/setup_qwen.py
.\.venv\Scripts\python.exe scripts/setup_qwen.py --download
```

O primeiro comando apenas confere os arquivos e o espaço. O segundo baixa Qwen 2.5 1.5B Instruct Q4_K_M e o pacote CPU do llama.cpp, verificando os hashes. O modelo selecionado ocupa aproximadamente 1,1 GB. Se o script indicar espaço insuficiente, libere espaço ou escolha outro disco conforme [GUIA_QWEN.md](GUIA_QWEN.md). Não avance se o download falhar.

**2. Iniciar o servidor em um terminal.**

```powershell
.\.venv\Scripts\python.exe scripts/run_qwen_server.py
```

Aguarde o servidor indicar que está pronto em `127.0.0.1:8081` e mantenha esse terminal aberto. Ctrl+C encerra o servidor. Não é necessário executar este comando novamente enquanto ele estiver funcionando.

**3. Validar o Qwen em um segundo terminal, na mesma pasta.**

```powershell
.\.venv\Scripts\python.exe scripts/verify_qwen.py
```

O esperado é `PASSOU` nos cinco casos: presente, negado, contraste, normal e incerto. A saída fica em `outputs/qwen_validation/results.json`; `all_passed` deve ser `true`. Se aparecer `FALHOU` ou `unavailable`, confira a saída e o terminal do servidor antes de seguir. Os cinco exemplos são uma validação de demonstração, não um benchmark geral ou validação clínica.

`FALHOU / ok` significa que o modelo retornou JSON válido, mas as categorias/afirmações não coincidiram com o esperado. `invalid_response` indica quebra do contrato, por exemplo evidência inexistente. O verificador mostra esperado/recebido e a razão de validação quando disponível. Os exemplos de instrução ficam separados dos cinco controles; os critérios não são ajustados para aceitar respostas erradas.

**4. Executar Whisper e Qwen juntos, mantendo o servidor aberto.**

```powershell
.\.venv\Scripts\python.exe -m src.main --audio data/samples/consulta_falada_sintetica.wav --speech-provider whisper --nlp-provider qwen --require-transcription --require-qwen --output-dir outputs/whisper_qwen --alert-db outputs/whisper_qwen/alerts.sqlite --prefix whisper_qwen
```

Os relatórios ficam em `outputs/whisper_qwen/`. No JSON recém-gerado, conferir `audio.stt_engine=whisper-local`, `fusion.cloud_nlp.engine=qwen-local` e `status=ok` nos dois módulos, com `model_inference_executed=true` e `cloud_executed=false`. `cloud_nlp` é o nome legado do campo, também usado para provedores locais. Os scores lexicais permanecem identificados separadamente.

Esse comando seleciona Qwen explicitamente; para torná-lo o padrão em execuções posteriores, altere `NLP_PROVIDER=qwen` no seu `.env` e inicie o servidor antes de rodar a CLI. Não é necessário preencher chaves Azure para esse caminho.

## Escolher Azure ou combinar provedores

Provedores: `--speech-provider whisper|azure|offline|legacy` e `--nlp-provider azure|local|aws|qwen`. Os padrões são Whisper e regras textuais locais; isso ainda não equivale a usar um LLM para texto. Nenhum fallback é apresentado como execução do serviço selecionado. OpenAI para relatório é opcional e exige `REPORT_PROVIDER=openai`; o padrão usa template local.

Azure Speech e Azure AI Language continuam disponíveis como alternativa, configurável no `.env` ou por flags. Também é possível combinar Whisper com Azure Language. Consulte [GUIA_AZURE.md](GUIA_AZURE.md). Com os recursos configurados, a prova dos dois serviços é:

```powershell
.\.venv\Scripts\python.exe scripts/verify_azure.py --audio data/samples/consulta_falada_sintetica.wav
```

Essa prova exige os dois serviços Azure; não carrega o modelo Whisper. Para o modo híbrido, use `--speech-provider whisper --nlp-provider azure` na CLI e não use `--require-azure`. Os verificadores carregam o `.env` da raiz, preservando valores definidos no ambiente do sistema.

Os relatórios, WAVs de trabalho e vídeos anotados recebem identificadores únicos para preservar execuções repetidas ou concorrentes.

## Testes e treino

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m pip check
```

Os testes usam provedores simulados/locais, sem credenciais reais. Cobrem negação, ausência de áudio, adaptação dos SDKs Azure, detectores temporais, unidades de dose, qualidade, isolamento por paciente, deduplicação e receptor HTTP.

Retreino opcional: `python scripts/prepare_sample_data.py` e `python scripts/train_yolo.py`. O treino resolve o dataset localmente e exporta métricas do peso final. Os dados YOLO são sintéticos.

## Arquivos de entrada e resultados

Os dados sintéticos em `data/samples/` são entradas para reproduzir os exemplos. Os testes em `tests/` verificam o comportamento do código. Os verificadores `scripts/verify_whisper.py`, `scripts/verify_qwen.py` e `scripts/verify_azure.py` exercitam os provedores reais quando preparados e configurados.

Relatórios, vídeos anotados, logs e bancos gerados ficam em `outputs/` e não precisam ser commitados. O `.gitignore` mantém a exceção original para `outputs/examples/`; novos resultados nessa pasta devem ficar fora do stage. Para avaliar o projeto, execute os comandos deste README e inspecione os resultados da sua execução. Para reproduzir automaticamente os cenários de monitoramento e verificar o receptor HTTP, execute:

```powershell
.\.venv\Scripts\python.exe scripts/validate_requirements.py
```

## Entregáveis

- [Relatório técnico](RELATORIO_TECNICO.md).
- Vídeo YouTube/Vimeo público ou não listado, até 15 minutos: **link disponível no PDF de entrega**.

Os dados do repositório são sintéticos. O protótipo não possui validação clínica nem representa um sistema hospitalar de produção.
