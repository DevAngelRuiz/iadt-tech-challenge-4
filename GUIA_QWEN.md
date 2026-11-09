# Qwen local — instalação e inferência real

## Estado verificável

O Qwen foi instalado e executado localmente em CPU. Após a correção do prompt e da geração de citações, os cinco controles originais passaram; em oito frases adicionais, seis passaram e duas falharam. A suíte automatizada passou com 88 testes, separadamente da avaliação do modelo. Consulte `RELATORIO_TECNICO.md` para resultados e limites. O modelo 1.5B pode omitir achados e confundir categorias: não há garantia de interpretação clínica. Quem clonar o repositório deve seguir a instalação e repetir as verificações. O provisionamento mantém 2 GiB de reserva e deve conferir novamente o espaço.

**Próxima ação prática: executar a sequência abaixo, conferindo primeiro o espaço disponível.** Não apagar arquivos pessoais ou ambientes sem verificar sua finalidade.

Escolhemos Qwen 2.5 1.5B Instruct em GGUF Q4_K_M, com llama.cpp para CPU no Windows. Não exige conta Azure, chave de API, Docker ou reinstalação do Ollama. A sugestão da família Qwen como alternativa ao Azure AI Language foi atribuída ao Prof. Jose Neto pela equipe; a escolha desta variante e do executor é uma decisão técnica do projeto.

## O que mudou

- `--nlp-provider qwen` chama somente um servidor em loopback, por padrão `http://127.0.0.1:8081/v1`.
- Retorna resumo, sentimento textual e achados com categoria, presença/negação/incerteza e evidência literal.
- Respostas incompletas, outro nome de modelo, esquema inválido e evidências ausentes do texto são rejeitados. A evidência literal não garante que a interpretação esteja correta.
- Os scores continuam calculados pelas regras locais, identificadas separadamente. Achados positivos do Qwen podem gerar `revisao_textual_llm`, de atenção, para revisão humana; achados negados e incertos não geram esse alerta.
- O relatório inclui a análise do Qwen. Se estiver indisponível, isso aparece no JSON e na qualidade da avaliação; não há sucesso simulado.
- `--require-qwen` retorna código 3 se não houver resposta de inferência válida. O provedor padrão continua `local` até concluir a instalação e validação.
- `verify_qwen.py` carrega o `.env` da raiz, assim como os verificadores Azure e Whisper; variáveis existentes no sistema têm prioridade. `QWEN_BASE_URL` e `QWEN_MODEL` podem ser configuradas nesse arquivo.
- O prompt define a taxonomia e apresenta exemplos completos, diferentes dos controles. A geração das citações é limitada a fragmentos literais da entrada; essa segmentação não decide categorias ou afirmações. `prompt_version` identifica a configuração utilizada.

## 1. Conferir e preparar arquivos

Na raiz do repositório, no PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/setup_qwen.py
.\.venv\Scripts\python.exe scripts/setup_qwen.py --download
```

O primeiro comando apenas confere o espaço. O segundo baixa SOMENTE se houver espaço suficiente. Modelo e programa ficam em `models/qwen/`, ignorados pelo Git. Revisões e SHA256 são fixados no script e conferidos antes de aceitar os arquivos. O pacote oficial do modelo selecionado tem 1.117.320.736 bytes (aproximadamente 1,1 GB); tamanhos de outros empacotamentos podem diferir.

Para armazenar o modelo e o executor em outro disco ou diretório, use `--directory` nos scripts de preparação e servidor, informando o mesmo caminho nos dois comandos.

## 2. Iniciar o servidor

```powershell
.\.venv\Scripts\python.exe scripts/run_qwen_server.py
```

Mantenha esse terminal aberto; Ctrl+C encerra o servidor. O processo usa CPU, duas threads, uma requisição por vez e contexto de 4096 tokens. Whisper, YOLO, Windows e Qwen compartilham a RAM; o desempenho depende dos recursos disponíveis e dos componentes executados simultaneamente.

## 3. Validar o modelo real em outro terminal

```powershell
.\.venv\Scripts\python.exe scripts/verify_qwen.py
```

A saída é salva em `outputs/qwen_validation/results.json`. São cinco casos sintéticos: relato positivo, negação, contraste, normalidade e incerteza. Cada caso exige coincidência exata entre os pares categoria/afirmação esperados e obtidos. Cinco casos não equivalem a validação clínica ou benchmark geral. Se falhar, inspecionar as saídas e corrigir antes de apresentar o componente como validado.

## 4. Executar Whisper e Qwen juntos

Somente após os passos anteriores:

```powershell
.\.venv\Scripts\python.exe -m src.main --audio data/samples/consulta_falada_sintetica.wav --speech-provider whisper --nlp-provider qwen --require-transcription --require-qwen --output-dir outputs/whisper_qwen --alert-db outputs/whisper_qwen/alerts.sqlite --prefix whisper_qwen
```

Conferir no JSON recém-gerado:

- `audio.stt_engine`: `whisper-local`; `audio.status`: `ok`.
- `fusion.cloud_nlp.engine`: `qwen-local`; `status`: `ok`.
- Os dois mecanismos devem ter `model_inference_executed: true` e `cloud_executed: false`.
- `fusion.cloud_nlp.served_model` corresponde ao alias configurado pelo servidor.
- `fusion.text_classification.model` continua `lexical-rules-with-negation`; não atribuir esses scores ao Qwen.

O nome legado `cloud_nlp` no JSON abriga também provedores locais; os metadados explicitam a execução local. O resumo gerado deve ser revisado antes de qualquer uso.

## Limitações

A execução real dos controles foi concluída em 07/11/2026. Os seis acertos nas oito frases adicionais não representam precisão geral do modelo. Resultados e erros observados estão em `RELATORIO_TECNICO.md`. A aplicação executa localmente; essa execução não comprova implantação em nuvem.

## Referências oficiais

- [Modelo Qwen e suporte a português](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
- [Distribuição oficial GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF).
- [Pacote llama.cpp fixado](https://github.com/ggml-org/llama.cpp/releases/tag/b12188).
- [Servidor e saídas estruturadas](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
