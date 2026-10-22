"""Server-side selection of the Stage 14 preview provider (Student 5).

The real generative provider is chosen only by the server environment, never by a tool payload:

    PREVIEW_PROVIDER=openai          the only switch for real image generation
    PREVIEW_API_KEY, PREVIEW_MODEL   required when PREVIEW_PROVIDER=openai
    PREVIEW_ALLOWED_IMAGE_HOSTS      optional exact host allow-list for HTTPS product images

Nothing is read at import time. `server_preview_provider()` reads `PREVIEW_PROVIDER` only when a
`preview` action runs:

- unset or empty: no provider, so the caller returns the normal product-card fallback;
- exactly `openai`: a lazy provider. The real `OpenAIImagePreviewProvider` (and its SDK
  client) is only created by `create_live_provider` when a generative preview is actually
  requested, i.e. after the PreviewRequest (including the portrait consent rule) has been
  validated;
- any other value: a provider that reports a configuration error, so the preview falls back
  with a configuration warning. The value itself is never echoed.

Error handling. Known, recoverable failures are `PreviewProviderError`s: the existing
`OpenAIImagePreviewProvider` already raises them for missing settings, SDK or Pillow
(`PreviewConfigurationError`), unusable inputs and network errors (`PreviewInputError`), SDK API
errors and invalid output. The only extra conversion here is an `openai.OpenAIError` raised
while the SDK client is constructed. `generate_preview_with_fallback` turns these into a
product-card fallback whose warning names no key, model, path or host. One server-side log line
records only a fixed category (configuration, input, output or provider) and the exception type
name, never the exception message, so no URL, path, environment value, portrait detail, model
output or API response can reach the log (an SDK error raised while the client is created stays
chained for debugging but is not logged). Any other exception
(AttributeError, TypeError, ...) is a programming error and propagates unchanged.

No `.env` file is read here: the deployment provides the environment variables.

All public names here are PROVISIONAL.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping

from travel_agent.recommendation.student5.models import PreviewRequest, PreviewResult
from travel_agent.recommendation.student5.preview_adapter import (
    OpenAIImagePreviewProvider,
    PreviewConfigurationError,
    PreviewInputError,
    PreviewOutputError,
    PreviewProvider,
    PreviewProviderError,
)

logger = logging.getLogger(__name__)

PREVIEW_PROVIDER_ENV = "PREVIEW_PROVIDER"
LIVE_PROVIDER_NAME = "openai"


def create_live_provider(environ: Mapping[str, str]) -> PreviewProvider:
    """PROVISIONAL. Create the real provider from PREVIEW_* variables.

    The single construction point for the real provider. Tests replace it with a fake.
    """
    return OpenAIImagePreviewProvider.from_env(environ)


def _error_category(exc: PreviewProviderError) -> str:
    """Fixed log category for a recoverable provider failure."""
    if isinstance(exc, PreviewConfigurationError):
        return "configuration"
    if isinstance(exc, PreviewInputError):
        return "input"
    if isinstance(exc, PreviewOutputError):
        return "output"
    return "provider"


def _is_sdk_error(exc: BaseException) -> bool:
    """True for an `openai.OpenAIError`; never imports the SDK itself."""
    sdk = sys.modules.get("openai")
    error_type = getattr(sdk, "OpenAIError", None)
    return isinstance(error_type, type) and isinstance(exc, error_type)


class ServerConfiguredPreviewProvider:
    """PROVISIONAL. Creates the real provider on its first generative request.

    `environ` is the server environment (normally `os.environ`); payload values never reach it.
    """

    provider_name = LIVE_PROVIDER_NAME

    def __init__(self, environ: Mapping[str, str]) -> None:
        self._environ = environ
        self._provider: PreviewProvider | None = None

    def __repr__(self) -> str:
        return "ServerConfiguredPreviewProvider()"

    def _live_provider(self) -> PreviewProvider:
        if self._provider is None:
            try:
                self._provider = create_live_provider(self._environ)
            except PreviewProviderError:
                raise
            except Exception as exc:
                if not _is_sdk_error(exc):
                    raise  # programming error: do not hide it behind a fallback
                # SDK messages may quote request details, so only the type is kept.
                raise PreviewConfigurationError(
                    f"the image SDK client could not be created ({type(exc).__name__})"
                ) from exc
        return self._provider

    def generate_preview(self, request: PreviewRequest) -> PreviewResult:
        # OpenAIImagePreviewProvider converts its known failures to PreviewProviderError;
        # anything else propagates.
        try:
            return self._live_provider().generate_preview(request)
        except PreviewProviderError as exc:
            # Only a fixed category and the exception type are logged, never str(exc): a
            # provider message could carry a URL, path, user text or API response.
            logger.warning(
                "Live preview unavailable; using product cards: category=%s type=%s",
                _error_category(exc),
                type(exc).__name__,
            )
            raise


class UnsupportedPreviewProvider:
    """PROVISIONAL. Stands in when PREVIEW_PROVIDER has a value other than `openai`."""

    provider_name = "unsupported"

    def generate_preview(self, request: PreviewRequest) -> PreviewResult:
        raise PreviewConfigurationError("PREVIEW_PROVIDER has an unsupported value")


def server_preview_provider(
    environ: Mapping[str, str] | None = None,
) -> PreviewProvider | None:
    """PROVISIONAL. The provider selected by the server environment, or None.

    Unset or empty: None (product-card fallback). Exactly `openai`: the real provider, created
    lazily. Any other value: `UnsupportedPreviewProvider` (fallback with a configuration
    warning). Read at call time, never at import time; the value is never logged or returned.
    """
    env = os.environ if environ is None else environ
    value = env.get(PREVIEW_PROVIDER_ENV) or ""
    if not value.strip():
        return None
    if value == LIVE_PROVIDER_NAME:
        return ServerConfiguredPreviewProvider(env)
    logger.warning("PREVIEW_PROVIDER has an unsupported value; using product cards")
    return UnsupportedPreviewProvider()
