FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOCAL_AGENT_WORKSPACE=/workspace \
    OLLAMA_URL=http://ollama:11434

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 agent

COPY pyproject.toml README.md ./
COPY local_agent ./local_agent

RUN pip install --no-cache-dir .

RUN mkdir -p /workspace && chown -R agent:agent /workspace /app
USER agent
WORKDIR /workspace
ENTRYPOINT ["local-agent"]
CMD ["doctor"]
