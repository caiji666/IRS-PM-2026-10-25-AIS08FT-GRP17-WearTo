"""TLS termination belongs to the deployment platform; never log request secrets."""

import os

bind = "0.0.0.0:" + os.environ.get("PORT", "8080")
workers = 1
threads = 4
timeout = 90
accesslog = None
errorlog = "-"
loglevel = "info"
