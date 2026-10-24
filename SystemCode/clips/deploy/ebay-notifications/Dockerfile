FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.3
COPY pyproject.toml uv.lock ./
ENV UV_PYTHON_DOWNLOADS=never
RUN uv sync --locked --no-dev --extra ebay-notifications
COPY travel_agent ./travel_agent
COPY deploy/ebay-notifications/gunicorn.conf.py ./gunicorn.conf.py
RUN useradd --uid 10001 --create-home app && mkdir -p /var/lib/ebay && chown app:app /var/lib/ebay
USER app
ENV EBAY_NOTIFICATION_DB=/var/lib/ebay/notifications.sqlite3
EXPOSE 8080
CMD ["/app/.venv/bin/gunicorn", "--config", "gunicorn.conf.py", "travel_agent.products.notifications.service:create_app()"]
