import json
import re
from typing import Any, Dict

import requests


class SummarizerService:
    def __init__(self, ollama_url: str, model_name: str):
        self.ollama_url = ollama_url
        self.model_name = model_name

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        raw_text = raw_text.strip()

        # 1) tentative directe
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # 2) extraire le premier bloc JSON
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 3) fallback
        return {
            "summary": raw_text if raw_text else "Aucun résumé généré.",
            "action_items": [],
            "completed_items": [],
            "important_points": [],
            "blockers": [],
            "keywords": [],
            "priority": "medium",
        }

    def summarize_structured(self, transcription: str) -> Dict[str, Any]:
        if not transcription or not transcription.strip():
            return {
                "summary": "Aucune transcription à résumer.",
                "action_items": [],
                "completed_items": [],
                "important_points": [],
                "blockers": [],
                "keywords": [],
                "priority": "low",
            }

        prompt = f"""
Tu es un assistant de productivité.
Analyse la transcription suivante en français.

Retourne UNIQUEMENT un JSON valide.
Sans markdown.
Sans explication.
Sans texte avant ou après.

Le format attendu est exactement :

{{
  "summary": "Résumé clair en 3 à 6 phrases",
  "action_items": ["tâche 1", "tâche 2"],
  "completed_items": ["élément 1", "élément 2"],
  "important_points": ["point 1", "point 2"],
  "blockers": ["blocage 1", "blocage 2"],
  "keywords": ["mot-clé 1", "mot-clé 2"],
  "priority": "low"
}}

Règles :
- priority doit être l'une de ces valeurs : low, medium, high
- si une liste est vide, retourne []
- reste fidèle à la transcription
- n'invente pas d'informations

Transcription :
\"\"\"
{transcription}
\"\"\"
        """.strip()

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
            },
            timeout=180,
        )
        response.raise_for_status()

        data = response.json()
        raw_output = data.get("response", "").strip()

        structured = self._extract_json(raw_output)

        # normalisation minimale
        structured.setdefault("summary", "")
        structured.setdefault("action_items", [])
        structured.setdefault("completed_items", [])
        structured.setdefault("important_points", [])
        structured.setdefault("blockers", [])
        structured.setdefault("keywords", [])
        structured.setdefault("priority", "medium")

        if structured["priority"] not in {"low", "medium", "high"}:
            structured["priority"] = "medium"

        return structured