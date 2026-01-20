FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --extra flows

COPY . .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "python"]
