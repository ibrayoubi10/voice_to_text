import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


def summarize_text(transcription: str, model: str = DEFAULT_MODEL) -> str:
    if not transcription or not transcription.strip():
        return "Aucune transcription à résumer."

    prompt = f"""
Tu es un assistant de productivité.

Résume la transcription suivante en français.
Retourne exactement ce format :

Résumé général :
...

Tâches réalisées :
- ...

Tâches à faire :
- ...

Points importants :
- ...

Blocages / problèmes :
- ...

Transcription :
\"\"\"
{transcription}
\"\"\"
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()