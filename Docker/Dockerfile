FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libegl1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

ENV MUJOCO_GL=disabled

CMD ["uv", "run", "pytest", "tests/", "-v"]
