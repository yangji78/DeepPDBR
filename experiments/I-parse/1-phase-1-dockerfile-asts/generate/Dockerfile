FROM python:3.8.18-slim-bookworm

RUN apt-get update \
  && apt-get install -y --no-install-recommends xz-utils \
  && rm -rf /var/apt/lists/*

RUN pip install --no-cache dockerfile tqdm

COPY . /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
