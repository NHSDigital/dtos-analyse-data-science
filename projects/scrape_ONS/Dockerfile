FROM python:3.12
LABEL version="0.1"
LABEL description="docker image for scrape_ons"

RUN pip install poetry

WORKDIR /app

COPY . .

RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "ons_client/download_filtered_csv.py"]
