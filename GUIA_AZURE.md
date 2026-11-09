# Configurar e comprovar os serviços Azure

**Provedor opcional:** após a orientação do Prof. Jose Neto informada pela equipe, o fluxo principal de áudio foi adaptado para Whisper local. Azure continua disponível para quem preferir Speech e Language; não é necessário criar recursos Azure para executar Whisper. Qwen também teve inferência local executada nos controles sintéticos; resultados e limites estão em `RELATORIO_TECNICO.md`. Consulte `GUIA_WHISPER.md` e `GUIA_QWEN.md` para instalar e executar o caminho local.

O código integra Azure Speech to Text e Azure AI Language (Text Analytics). A integração real permanece pendente até criar os recursos, configurar as credenciais e executar um áudio com fala. Testes com SDKs simulados não comprovam uso da nuvem.

## Criar os recursos

1. Entre no [portal Azure](https://portal.azure.com/) com sua conta e escolha sua assinatura e grupo de recursos.
2. Crie um recurso **Speech**. Anote a região e obtenha uma chave em **Keys and Endpoint / Chaves e ponto de extremidade**.
3. Crie um recurso **Azure AI Language** com as funcionalidades de análise de texto. Obtenha a chave e o endpoint na página do recurso.
4. Confira o plano e as condições da assinatura antes de criar os recursos. Use a camada gratuita quando ela estiver disponível e atender ao cenário; o código não provisiona recursos nem altera sua assinatura.

Referências oficiais: [Speech to Text em Python](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-speech-to-text?pivots=programming-language-python) e [SDK de Text Analytics](https://learn.microsoft.com/python/api/overview/azure/ai-textanalytics-readme?view=azure-python).

## Configurar localmente

Preencha o `.env` na raiz, sem enviar as chaves pelo chat e sem versionar o arquivo. O `.env.example` contém os nomes das variáveis; não substitua um `.env` já preenchido ao copiar o exemplo.

```dotenv
AZURE_SPEECH_KEY=sua_chave_speech
AZURE_SPEECH_REGION=regiao_do_recurso_speech
AZURE_LANGUAGE_KEY=sua_chave_language
AZURE_LANGUAGE_ENDPOINT=https://seu-recurso.cognitiveservices.azure.com/
SPEECH_PROVIDER=azure
NLP_PROVIDER=azure
REPORT_PROVIDER=local
```

O endpoint acima é apenas um formato de exemplo: copie o valor exato do portal. Speech e Language podem ter chaves diferentes. Nenhuma chave é impressa no comando de verificação.

Instale as dependências:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest
.\.venv\Scripts\python.exe scripts/verify_azure.py
```

O segundo comando só verifica a presença da configuração. Ele não prova que as chaves são válidas.

## Preparar a demonstração de fala

Grave um áudio curto, autorizado e sem dados pessoais, por exemplo: “Estou muito cansada. Sinto falta de ar quando caminho. Gostaria de conversar com a equipe.” Use WAV PCM de 16 bits; outros formatos são convertidos pelo MoviePy quando suportados. O arquivo `consulta_hesitante.wav` do repositório contém tons, não uma consulta falada, e não serve como evidência de STT.

```powershell
.\.venv\Scripts\python.exe scripts/verify_azure.py --audio caminho/consulta_falada.wav
```

Esse comando executa a CLI com `--require-azure`, sem transcrição manual. Ele deve terminar com código 0 e gerar JSON/Markdown em `outputs/azure_validation/`. No JSON, conferir:

- `audio.stt_engine`: `azure-speech`;
- `audio.status`: `ok`, com transcrição não vazia;
- `fusion.cloud_nlp.engine`: `azure-text-analytics`;
- `fusion.cloud_nlp.status`: `ok`;
- `cloud_executed`: `true` nos dois provedores.

O relatório mostra sentimento, entidades e frases-chave retornados pelo Azure no JSON completo. Os termos de risco também são identificados por regras lexicais locais explicitamente nomeadas. O sentimento do texto não é interpretado como diagnóstico.

O rótulo de sentimento de um documento é preservado, inclusive `MIXED`. Textos longos são particionados; `sentiment_documents` registra rótulo, scores e posição de cada trecho. A agregação resulta em `MIXED` quando houver trecho misto ou trechos positivos e negativos; nos demais casos preserva o rótulo não neutro ou `NEUTRAL`. `sentiment_aggregation` identifica o método. Os scores agregados são médias ponderadas pelo tamanho dos trechos, não probabilidades clínicas.

Os verificadores carregam o `.env` da raiz sem substituir variáveis já definidas no sistema. As flags da CLI prevalecem sobre os padrões dos provedores. Para combinar Whisper e Language, execute a CLI com `--speech-provider whisper --nlp-provider azure`; o modo `--require-azure` fica reservado à comprovação dos dois serviços Azure.

## Falhas e evidências

- Código de saída **3**: Azure não foi comprovado (configuração ausente, falha, timeout ou ausência de fala).
- `not_configured`: faltam chave/região/endpoint; não houve chamada ao serviço.
- `dependency_missing`: instale os SDKs de `requirements.txt` no mesmo Python usado na execução.
- `error`: confira a região, o endpoint, as permissões e a conectividade. A saída registra o tipo de falha, sem copiar mensagens que possam expor informações da conta.
- `no_speech`: o serviço não reconheceu fala. Isso não comprova ausência de risco clínico.
- `manual`: houve `--transcript-override`, que não é aceito como evidência no modo estrito.

O relatório local pode usar template. A execução do Azure AI Language é identificada pelos metadados do serviço de análise de texto, independentemente do uso opcional de OpenAI para geração de relatório.
