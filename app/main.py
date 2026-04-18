from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from ollama import ResponseError as OllamaResponseError
except ImportError:  # pragma: no cover

    class OllamaResponseError(Exception):
        pass


from app.graph import run_agent
from app.ollama_ready import wait_for_ollama_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    wait_for_ollama_model()
    yield


app = FastAPI(
    title="LocalScript API",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Текст задачи на естественном языке")


@app.post(
    "/generate",
    response_model=dict[str, str],
    response_model_exclude_unset=True,
)
def generate(req: GenerateRequest) -> dict[str, str]:
    """Ответ — JSON-объект с 1+ парами: `{\"<тег>\": \"lua{...весь код...}lua\", ...}` на корне JSON."""
    try:
        bindings = run_agent(req.prompt)
    except OllamaResponseError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        msg = str(e).lower()
        if "memory" in msg or "requires more system memory" in msg:
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise
    if not bindings:
        raise HTTPException(
            status_code=422,
            detail="Model did not return valid LowCode JSON (one or more tags with lua{...}lua values) after validation.",
        )
    return bindings


@app.get("/health")
def health():
    return {"status": "ok"}
