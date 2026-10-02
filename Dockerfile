# Three stages off one base so the tests run against exactly what ships:
#   base    -> dependencies + application code
#   test    -> base + pytest + tests/      (docker compose run --rm --build test)
#   runtime -> base + the server command   (docker compose up) -- the default
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements.txt -r requirements-api.txt

# Runtime settings go BELOW the install above, so changing one never throws
# away the cached dependency layer.
# One numeric thread: the scikit-learn work here is on tiny datasets, where
# OpenBLAS/OpenMP fanning out across every core is ~10x SLOWER than a single
# thread (measured on the test suite), and would crowd a shared VPS.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Code stays root-owned: the app user can read and run it, not rewrite it.
COPY agent/ agent/
COPY web/ web/
COPY api.py main.py ./
# Seed audit history -- the one path the app user has to write to. A named
# volume mounted at /app/data is filled from this on first start only, so
# later deploys never overwrite the live database.
COPY --chown=app:app data/ data/


FROM base AS test

COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
COPY tests/ tests/
COPY pytest.ini ./
# pytest writes its cache into the working directory.
RUN chown app:app /app
USER app
CMD ["pytest"]


FROM base AS runtime

# Refuse to fetch loopback/private/link-local addresses (agent/netguard.py).
# Off by default for CLI use; a publicly reachable server must have it on.
ENV SEO_AGENT_BLOCK_PRIVATE_HOSTS=1 \
    SEO_AGENT_DB_PATH=/app/data/audit_history.db

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# Exactly one worker: api.py keeps running jobs in process memory.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
