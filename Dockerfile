FROM python:3.10

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

CMD gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 2 --timeout 120 LesChatsDeSeatech:app
