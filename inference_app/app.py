from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from model_loader import LoadedModel, load_artifact, predict_text


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    input_text: str
    output_text: str
    model_kind: str
    checkpoint_path: str


def create_app(loaded: LoadedModel) -> FastAPI:
    app = FastAPI(title="Arabic Tashkeel Inference")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Arabic Tashkeel Demo</title>
  <style>
    body {{ font-family: Segoe UI, sans-serif; margin: 2rem; background: #f6f8fb; color: #18212f; }}
    .card {{ max-width: 900px; margin: 0 auto; background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); }}
    textarea {{ width: 100%; min-height: 140px; padding: 0.9rem; font-size: 1.05rem; border-radius: 10px; border: 1px solid #c8d2e0; box-sizing: border-box; }}
    button {{ margin-top: 1rem; background: #0f766e; color: white; border: 0; padding: 0.8rem 1.2rem; border-radius: 10px; cursor: pointer; font-size: 1rem; }}
    button:hover {{ background: #115e59; }}
    .meta {{ margin-bottom: 1rem; color: #475569; }}
    .result {{ margin-top: 1.5rem; padding: 1rem; background: #f1f5f9; border-radius: 10px; white-space: pre-wrap; direction: rtl; font-size: 1.2rem; }}
    .error {{ color: #b91c1c; margin-top: 1rem; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Arabic Tashkeel Demo</h1>
    <div class="meta">
      Model: <strong>{loaded.model_kind}</strong><br>
      Checkpoint: <code>{loaded.checkpoint_path}</code><br>
      Device: <code>{loaded.device}</code>
    </div>
    <label for="text">Undiacritized Arabic text</label>
    <textarea id="text" placeholder="اكتب النص العربي هنا"></textarea>
    <button onclick="runPredict()">Predict</button>
    <div id="error" class="error"></div>
    <div id="result" class="result"></div>
  </div>
  <script>
    async function runPredict() {{
      const text = document.getElementById('text').value;
      const errorEl = document.getElementById('error');
      const resultEl = document.getElementById('result');
      errorEl.textContent = '';
      resultEl.textContent = '';
      try {{
        const response = await fetch('/api/predict', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ text }})
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.detail || 'Prediction failed');
        }}
        resultEl.textContent = data.output_text;
      }} catch (error) {{
        errorEl.textContent = error.message;
      }}
    }}
  </script>
</body>
</html>"""

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text must not be empty.")
        output_text = predict_text(loaded, text)
        return PredictResponse(
            input_text=text,
            output_text=output_text,
            model_kind=loaded.model_kind,
            checkpoint_path=str(loaded.checkpoint_path),
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a FastAPI inference app for Arabic tashkeel.")
    parser.add_argument("--artifact-dir", required=True, help="Path to artifact directory containing checkpoint/config/vocab.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_artifact(Path(args.artifact_dir), device=args.device)
    app = create_app(loaded)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
