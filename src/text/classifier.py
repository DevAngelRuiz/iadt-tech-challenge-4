"""Regras lexicais com negação; não é um classificador clínico treinado."""
import re
import unicodedata
from src.config import RISK_LEXICON


def _normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn")


# Reinicia o escopo em uma oração coordenada com predicado próprio, mas não
# em listas como "nega violência e ameaça" ou "sem falta de ar e disartria".
_PREDICATE_BOUNDARY = re.compile(
    r"(?:,\s*|\be\s+)(?:(?:eu|ela|ele|a paciente|o paciente|paciente)\s+)?"
    r"(?:(?:nao|tambem)\s+)*"
    r"(?:estou|esta|estava|sinto|sente|sentia|tenho|tem|tinha|relato|relata|"
    r"refere|apresenta|sofro|sofre|consigo|consegue|"
    r"me\s+(?:bate|machuca|controla|ameaca))\b"
)


def _is_negated(clause: str, start: int, end: int, intrinsic_phrases: set[str]) -> bool:
    boundary = 0
    # Incluir a expressão atual permite reconhecer "e ele me bate", mesmo
    # quando o predicado faz parte do próprio termo procurado.
    for match in _PREDICATE_BOUNDARY.finditer(clause[:end]):
        if match.start() < start:
            boundary = match.start()
    prefix = clause[boundary:start]
    for phrase in intrinsic_phrases:
        prefix = prefix.replace(phrase, " ")
    scope = " ".join(prefix.split()[-8:])
    return bool(re.search(r"\b(nega|negam|negou|nao|sem|ausencia de)\b", scope))


def classify_clinical_text(text: str) -> dict:
    text = text or ""
    clauses = re.split(r"[.;!?\n]|\b(?:mas|porem|contudo|entretanto)\b", _normalize(text or ""))
    hits, negated = {}, {}
    lexicon = {key: set(_normalize(p) for p in phrases) for key, phrases in RISK_LEXICON.items()}
    lexicon["depressao_pos_parto"].update({"choro", "depressao"})
    lexicon["ansiedade"].add("medo")
    lexicon["respiratorio"] = {"falta de ar", "dificuldade para respirar", "dificuldade respiratoria"}
    lexicon["dificuldade_fala"] = {"fala arrastada", "dificuldade para falar", "disartria"}
    intrinsic_phrases = {p for phrases in lexicon.values() for p in phrases if p.startswith(("nao ", "sem "))}
    for label, phrases in lexicon.items():
        for phrase in sorted(phrases):
            for clause in clauses:
                for match in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", clause):
                    denied = _is_negated(clause, match.start(), match.end(), intrinsic_phrases)
                    target = negated if denied else hits
                    target.setdefault(label, set()).add(phrase)
    labels = {key: round(min(1.0, .35 * len(value)), 3) for key, value in hits.items()}
    risk = max(labels.values(), default=0.0)
    if labels.get("violencia", 0):
        risk = min(1.0, max(risk, labels["violencia"] + .15))
    top = max(labels, key=labels.get) if labels else ("normal" if text.strip() else "indeterminado")
    return {"labels": labels or ({"normal": 1.0} if text.strip() else {}), "top_label": top,
            "text_risk_score": round(risk, 3), "hits": {k: sorted(v) for k, v in hits.items()},
            "negated_hits": {k: sorted(v) for k, v in negated.items()}, "model": "lexical-rules-with-negation"}
