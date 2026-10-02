# Producer image.
#
# The producer runs on the host by default (`make produce`) because that is the
# fastest edit-test loop and the metrics endpoint is scraped directly by
# Prometheus. This image exists for the two cases where host execution is not
# available: a self-contained demo where nothing is installed beyond Docker, and
# a load generator that should not compete with the developer's own processes.
#
# Build context is the repository root, because the producer imports the shared
# `common/` package. Keep the context small with .dockerignore.

FROM python:3.11-slim AS base

# Byte-code is written at build time, not run time, so the first micro-batch in
# a fresh container is not slowed by compilation. Unbuffered output so log lines
# appear immediately rather than at process exit.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, as their own layer: they change far less often than source,
# so a source edit does not trigger a reinstall.
COPY producer/requirements.txt /app/producer/requirements.txt
RUN pip install --no-cache-dir -r /app/producer/requirements.txt

# Application code. `common/` is shared, so it is copied alongside the producer.
COPY common/ /app/common/
COPY producer/ /app/producer/

# Run as an unprivileged user. The producer only needs to read the mounted CSV
# and open its metrics port, so no part of this needs root.
RUN useradd --create-home --uid 10001 producer
USER producer

# Metrics endpoint. Compose maps the host port; the in-network address is
# <service>:8001.
EXPOSE 8001

# Default: stream the mounted dataset at the configured rate. Override the
# command to pass flags, e.g. `docker run ... --tps 5000 --max-records 100000`.
ENTRYPOINT ["python", "-m", "producer.simulate_stream"]
CMD ["--loop"]
