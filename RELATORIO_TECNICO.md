# Relatório técnico — Tech Challenge Fase 4

## Escopo e conformidade

Protótipo de análise multimodal para saúde da mulher, comparado ao PDF da Fase 4. O enunciado pede análise de vídeo, áudio e detecção de anomalias; não estabelece escolha de duas funcionalidades ou três objetivos.

### Adaptação tecnológica indicada na comunidade

Conforme orientação do **Prof. Jose Neto**, informada pela equipe a partir da comunidade do curso, foram indicados **Whisper** como alternativa ao Azure Speech e **Llama 3.1/3.2 ou Qwen 2.5** como alternativas ao Azure AI Language. Adotamos Whisper local e Qwen 2.5 1.5B, ambos executados sobre dados sintéticos. Qwen extrai resumo, sentimento e achados presentes/negados/incertos com evidência literal, separados dos scores lexicais. Após o ajuste da integração, os cinco controles originais passaram; em oito frases adicionais ocorreram seis acertos e duas falhas. A suíte automatizada passou com **88 testes**, separadamente da avaliação do modelo. Consulte a seção de limitações abaixo e o [guia Qwen](GUIA_QWEN.md) para reprodução. Azure permanece opcional, sem execução real nesta entrega.

O fluxo local executa os modelos no computador do usuário, sem serviço gerenciado em nuvem. A orientação tecnológica informada pela equipe não constitui validação dos resultados do protótipo.

| Requisito | Implementação | Evidência / estado |
|---|---|---|
| Vídeo e eventos anormais | YOLOv8 customizado, classe bleeding, anotações | Executado nos exemplos sintéticos positivo/negativo |
| Relatórios automáticos | JSON e Markdown, evidências e qualidade | CLI multimodal |
| Áudio de consulta | PCM mono 16 kHz, Whisper local, medidas vocais | Verificação pelo modo estrito de transcrição; dados de demonstração sintéticos |
| Azure Speech to Text | src/cloud/azure_services.py | SDK e testes simulados; recurso não configurado |
| Azure Text Analytics | Sentimento, entidades, frases-chave em português | SDK e testes simulados; recurso não configurado |
| Sinais vitais | Limites configurados e referência móvel de mediana/MAD | Cenários normais/anômalos de telemetria sintética |
| Prescrições | Comparação temporal de dose/frequência, inclusão e retirada | Cenários sintéticos, normalização mg/g/mcg |
| Movimentação | Atividade temporal e imobilidade persistente | Telemetria sintética com presença e qualidade explícitas |
| Alertas à equipe | Receptor SQLite e painel local com confirmação | Monitor publica antes do final do fluxo |
| Monitoramento progressivo | Replay e acompanhamento de JSONL em --follow | Demonstração local; não há conexão com sensores hospitalares |
| Vídeo até 15 minutos | Apresentação do projeto | Link ainda não disponibilizado no README |

O estado “implementado” não substitui evidência de execução. Whisper foi exercitado sobre fala sem override, e Qwen foi executado nos controles documentados, com seus erros preservados. As linhas Azure acima descrevem os provedores opcionais preservados, e não execução real desses serviços. A execução local não comprova implantação em nuvem.

## Fluxo multimodal

```text
Vídeo -> YOLO / emoção / pose -----------+
Áudio -> extração -> Whisper ou Azure ---+-> classificação + fusão -> JSON / Markdown
Texto -> regras / Qwen / Azure Language -+                         -> receptor local
Telemetria -> detectores online -> receptor SQLite -> painel -> confirmação de leitura
```

A CLI também aceita CSVs das três modalidades estruturadas e inclui seus alertas na fusão. A análise audiovisual ocorre em arquivos; o monitor estruturado processa registros progressivamente e pode acompanhar um produtor que acrescente linhas ao JSONL.

## Modelos e técnicas

### Vídeo

Pesos em `models/bleeding_yolov8n.pt`, treinados em 40 imagens sintéticas de treino e 10 de validação. Há fallback HSV identificado no JSON; ele não comprova execução do YOLO. Emoção usa DeepFace quando instalado ou heurística explicitamente identificada. Postura exige MediaPipe e visibilidade suficiente de ombros, cotovelos e punhos. Sem modelo/articulações visíveis, a postura fica indisponível e a avaliação é parcial. OpenCV calcula somente atividade global por diferença de frames, sem inferir postura ou movimentos dos braços.

As métricas versionadas em `models/bleeding_metrics.json` pertencem ao treinamento original. Elas não foram novamente estimadas nesta implementação. O script de treino foi ajustado para validar o peso entregue e exportar métricas e hash do peso. As métricas sintéticas não demonstram generalização clínica.

O campo `coverage` informa a fração de frames processados. A CLI passa limite nulo por padrão; eventuais limites são explícitos.

### Áudio e texto

Extração de áudio converte WAV PCM de 16 bits para mono/16 kHz; MoviePy atende outros formatos suportados. Falhas de extração não criam silêncio artificial. Uma transcrição manual é marcada como `manual`.

Whisper local recebe o vetor de áudio normalizado, usa português como idioma padrão e registra modelo, dispositivo, segmentos e inferência executada. Segmentos com baixa qualidade são filtrados. Essa filtragem não elimina erros ou alucinações. A implementação Azure opcional usa reconhecimento contínuo e Azure Language; nenhuma chamada Azure é necessária para o caminho Whisper com regras locais.

Risco lexical é uma regra local com normalização de acentos, limites de palavra e escopo de negação. Orações coordenadas com predicado próprio reiniciam esse escopo; listas negadas preservam a negação compartilhada. Retorna termos encontrados e negados. A regra continua heurística, sem compreensão linguística geral; não é aprendizagem supervisionada nem diagnóstico. Há categorias de sofrimento emocional, ansiedade, violência, fadiga, relatos respiratórios e dificuldade de fala.

Energia e pausas são características acústicas, não diagnósticos. O reforço acústico exige transcrição automática não vazia; transcrição manual não comprova fala. Pausas isoladas não geram ansiedade. O WAV de tons é um controle de ausência de fala.

### Anomalias temporais

Os detectores mantêm estado separado por paciente e variável. Timestamps precisam de fuso; ordem temporal, valores finitos e unidades são verificados.

- Sinais vitais: limites configuráveis e desvio robusto em relação à mediana/MAD dos valores anteriores. São necessários pelo menos cinco valores para a referência estatística; leituras anômalas não a contaminam.
- Prescrições: compara dose e frequência ao registro anterior, converte mg/g/mcg e identifica inclusão/retirada inesperada. Uma mudança planejada exige justificativa. O primeiro registro define referência; não há verificação farmacológica.
- Movimentação: requer paciente presente, qualidade suficiente e atividade baixa durante o período configurado. Ausência de paciente, baixa qualidade ou lacunas longas reiniciam a contagem.

Os limites do cenário em `monitor_rules.json` são demonstrativos, não um protocolo clínico. Os CSVs de movimento não são derivados dos vídeos de sangramento.

## Fusão e alertas

Os pesos originais são sangramento 0.35, emoção 0.20, postura 0.10, áudio 0.25 e texto 0.10. Níveis agregados são INFO, ATENCAO e CRITICO; a gravidade dos alertas tipados pode elevar o nível mesmo com score ponderado baixo.

Alertas estruturados são incluídos na decisão. O receptor registra paciente, origem, instante do dado, evidência, recebimento, execução e confirmação de leitura. Há deduplicação com janela de 60 segundos por episódio/execução, permitindo aumento para CRITICO.

O campo de latência mede detecção e preparação até o enfileiramento local. Não representa latência de ponta a ponta hospitalar. O painel faz consulta a cada segundo e exibe alertas enquanto o produtor ainda está processando.

Modalidades indisponíveis são registradas em `data_quality` e na cobertura. Um resultado INFO parcial não comprova ausência de risco. Credenciais faltantes não ativam um serviço alternativo silenciosamente.

## Reprodução e evidências

Os comandos completos estão no README. Gerar os dados com `scripts/prepare_monitoring_data.py`, iniciar `src.alert_server` e executar `src.monitor` com os controles normal e anômalo. Para uso integrado, a CLI aceita `--vitals`, `--prescriptions` e `--movement`.

Os resultados devem ser reproduzidos pelos comandos do README. Saídas geradas em `outputs/` não são incluídas no versionamento.

Os testes de adaptação Azure usam serviços simulados. O teste do receptor verifica que um alerta pode ser consultado e confirmado antes do próximo registro do produtor. O script `scripts/validate_requirements.py` reproduz os cenários e gera resultados locais em `outputs/requirements_validation/` e `outputs/examples/requisitos/`. Esses novos resultados ficam fora do commit.

## Limitações

A avaliação usa dados sintéticos e um conjunto pequeno de controles. O Qwen apresentou omissões e confusão de categorias; resultados estruturados e resumos requerem revisão humana. Nos casos adicionais, omitiu a negação de cansaço em “Nega cansaço. Apresenta fala arrastada.” e classificou medo como sofrimento emocional em “Sinto medo, mas não há ameaça.”. No fluxo integrado, o resumo mencionou falta de ar, mas o achado respiratório foi omitido na lista estruturada. Esses casos limitados não medem a precisão geral do modelo. Exemplos com transcrição manual estão identificados separadamente das execuções reais de Whisper. Modelos, limites e características acústicas não foram validados para aplicação clínica.

O sistema é um protótipo local com dados sintéticos; não há autenticação hospitalar, integração com sensores reais nem validação de diagnóstico.
