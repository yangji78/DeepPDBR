FROM python:3.8.18-slim-bookworm

RUN apt-get update \
  && apt-get install -y --no-install-recommends xz-utils \
  && rm -rf /var/apt/lists*

RUN pip install --no-cache tqdm

COPY . /app/

ENTRYPOINT [ "python3", "/app/app.py" ]
