# Dados sintéticos de monitoramento

Gerados por `python scripts/prepare_monitoring_data.py`. Não contêm dados reais de pacientes nem prescrições terapêuticas.

- `normal.jsonl`: fluxo normal misto; não deve gerar alertas.
- `anomalo.jsonl`: sinais vitais fora dos limites configurados, aumento não planejado de dose e baixa atividade persistente.
- Os CSVs separam as mesmas modalidades para a CLI multimodal.

Campos comuns: `patient_id` (identificador sintético) e `timestamp` ISO 8601 com fuso. Os registros devem estar em ordem temporal por paciente/modalidade. Dose e frequência devem ser positivas; valores não finitos e unidades desconhecidas são rejeitados.

Sinais vitais: `heart_rate_bpm`, `spo2_pct`, `systolic_mmhg`, `diastolic_mmhg`. No JSONL ficam dentro de `metrics`. Ausência de uma medida é registrada como cobertura incompleta, sem substituir por zero.

Prescrições: `medication`, `dose`, `unit` (`mg`, `g` ou `mcg`), `frequency_per_day`, `action` (`set`/`stop`), `planned_change` e `reason`. A primeira prescrição de cada paciente define uma referência. Mudanças planejadas precisam de justificativa. Os medicamentos têm nomes fictícios; são comparadas alterações estruturais, não adequação terapêutica.

Movimentação: `activity_score` entre 0 e 1, `patient_present` e `quality_good`. São medidas simuladas de atividade, não valores extraídos dos vídeos de sangramento. Sem paciente ou sem qualidade, o período de imobilidade é reiniciado e a indisponibilidade é registrada. Lacunas longas também reiniciam a contagem.

Os limites em `../monitor_rules.json` servem apenas à demonstração acadêmica. A técnica de mediana/MAD compara cada valor com o histórico anterior do mesmo paciente e da mesma variável; valores anômalos não entram nessa referência.
