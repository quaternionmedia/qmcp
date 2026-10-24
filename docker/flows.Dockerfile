FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
# vox is a path dependency (`[tool.uv.sources]` in pyproject.toml): the sync
# builds it from vendor/vox, so it is copied before the sync runs.
COPY vendor/vox ./vendor/vox
COPY qmcp ./qmcp
RUN uv sync --frozen --extra flows

COPY . .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "python"]
