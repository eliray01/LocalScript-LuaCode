import os

from langchain_ollama import ChatOllama


def get_llm() -> ChatOllama:
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    num_ctx = int(os.environ.get("NUM_CTX", "4096"))
    num_predict = int(os.environ.get("NUM_PREDICT", "256"))
    return ChatOllama(
        model=model,
        base_url=base_url,
        num_ctx=num_ctx,
        num_predict=num_predict,
        temperature=0.1,
    )
