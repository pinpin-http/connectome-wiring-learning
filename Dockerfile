# CPU-only image to check the preprint's numbers: hashes, tables, figures, statistics audit, unit tests.
# It does not train (training needs a GPU: README.md, deployment/README.md) and holds no ROM and no data.
#
#   docker build -t flymario .
#   docker run --rm flymario                                  # verify hashes + unit tests
#   docker run --rm -v "$PWD/artifacts:/repo/artifacts" flymario make figures audit
FROM python:3.11.16-slim

RUN apt-get update && apt-get install -y --no-install-recommends make poppler-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY requirements.txt pyproject.toml README.md ./
RUN python -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && .venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
COPY . .
RUN .venv/bin/pip install --no-cache-dir --no-deps -e .
CMD ["sh", "-c", "python tools/verify_results.py && .venv/bin/python -m pytest tests -q"]
