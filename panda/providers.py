"""Lazy, role-specific text provider boundary for planning and reflection.

Importing this module performs no client construction, network access, or writes.
Secrets are retained only inside resolved client configuration and are never included
in diagnostics or provider errors.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping
from urllib.parse import urlparse

_SUPPORTED_PROVIDERS = frozenset({"gemini", "openai"})
_DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class ProviderConfigurationError(ValueError):
    """Raised when a role's provider configuration is absent or unsafe."""


class ProviderRequestError(RuntimeError):
    """Raised when a provider call or response extraction fails safely."""


@dataclass(frozen=True)
class ProviderConfig:
    role: str
    provider: str
    model: str
    api_key: str = field(repr=False)
    base_url: str | None = None
    effective_source: Mapping[str, str | None] = field(
        default_factory=dict, compare=False
    )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "effective_source": dict(self.effective_source),
        }


ClientFactory = Callable[[ProviderConfig], Any]
Transport = Callable[[ProviderConfig, str, str], Any]
_CLIENT_CACHE: dict[tuple[Any, ...], Any] = {}


def clear_provider_client_cache() -> None:
    """Clear cached clients so subsequent calls use the current configuration."""
    _CLIENT_CACHE.clear()


def _required(value: str | None, variable: str, role: str) -> str:
    if value is None or not value.strip():
        raise ProviderConfigurationError(
            f"{variable} is required for the {role} provider"
        )
    return value.strip()


def _is_loopback_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _openai_value(
    environ: Mapping[str, str], role: str, suffix: str
) -> tuple[str | None, str]:
    role_name = "PLANNER" if role == "planning" else "REFLECTION"
    variable = f"{role_name}_{suffix}"
    if variable in environ:
        return environ.get(variable), variable
    if role == "reflection":
        fallback = f"PLANNER_{suffix}"
        return environ.get(fallback), fallback
    return environ.get(variable), variable


def resolve_provider_config(
    role: str,
    *,
    provider: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProviderConfig:
    """Resolve and validate one role without constructing a provider client."""
    if role not in {"planning", "reflection"}:
        raise ProviderConfigurationError(f"Unsupported provider role: {role!r}")
    env = os.environ if environ is None else environ
    provider_variable = (
        "PLANNER_PROVIDER" if role == "planning" else "REFLECTION_PROVIDER"
    )
    if provider is not None:
        selected = provider.strip().lower()
        provider_source = "argument"
    elif provider_variable in env:
        selected = str(env.get(provider_variable, "")).strip().lower()
        provider_source = provider_variable
    else:
        selected = "gemini"
        provider_source = "default"
    if not selected:
        raise ProviderConfigurationError(
            f"{provider_variable} is required for the {role} provider"
        )
    if selected not in _SUPPORTED_PROVIDERS:
        raise ProviderConfigurationError(
            f"Unsupported {role} provider {selected!r}; choose gemini or openai"
        )

    if selected == "gemini":
        api_key = _required(env.get("GEMINI_API_KEY"), "GEMINI_API_KEY", role)
        model = _required(
            env.get("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL), "GEMINI_MODEL", role
        )
        return ProviderConfig(
            role=role,
            provider=selected,
            model=model,
            api_key=api_key,
            base_url=None,
            effective_source={
                "provider": provider_source,
                "model": "GEMINI_MODEL" if "GEMINI_MODEL" in env else "default",
                "api_key": "GEMINI_API_KEY",
                "base_url": None,
            },
        )

    base_url_value, base_url_source = _openai_value(env, role, "BASE_URL")
    model_value, model_source = _openai_value(env, role, "MODEL")
    api_key_value, api_key_source = _openai_value(env, role, "API_KEY")
    base_url = _required(base_url_value, base_url_source, role).rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProviderConfigurationError(
            f"{base_url_source} must be an absolute HTTP(S) URL"
        )
    model = _required(model_value, model_source, role)
    api_key = _required(api_key_value, api_key_source, role)
    if api_key == "EMPTY" and not _is_loopback_url(base_url):
        raise ProviderConfigurationError(
            f"{api_key_source}=EMPTY is allowed only for a localhost/loopback base URL"
        )
    return ProviderConfig(
        role=role,
        provider=selected,
        model=model,
        api_key=api_key,
        base_url=base_url,
        effective_source={
            "provider": provider_source,
            "model": model_source,
            "api_key": api_key_source,
            "base_url": base_url_source,
        },
    )


def _default_client_factory(config: ProviderConfig) -> Any:
    if config.provider == "gemini":
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderConfigurationError(
                "google-genai is required for the gemini provider"
            ) from exc
        return genai.Client(api_key=config.api_key)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ProviderConfigurationError(
            "openai is required for the openai-compatible provider"
        ) from exc
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def _get_client(config: ProviderConfig, factory: ClientFactory) -> Any:
    key = (
        id(factory),
        config.role,
        config.provider,
        config.model,
        config.base_url,
        config.api_key,
    )
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = factory(config)
    return _CLIENT_CACHE[key]


def _extract_text(config: ProviderConfig, response: Any) -> str:
    if isinstance(response, str):
        text = response
    elif config.provider == "gemini":
        text = getattr(response, "text", None)
        if text is None and isinstance(response, Mapping):
            text = response.get("text")
    else:
        try:
            choice = (
                response["choices"][0]
                if isinstance(response, Mapping)
                else response.choices[0]
            )
            message = (
                choice["message"] if isinstance(choice, Mapping) else choice.message
            )
            text = (
                message.get("content")
                if isinstance(message, Mapping)
                else message.content
            )
        except (AttributeError, IndexError, KeyError, TypeError):
            text = None
    if not isinstance(text, str) or not text.strip():
        raise ProviderRequestError(
            f"{config.role} {config.provider} provider returned no text"
        )
    return text


def request_json_text(
    *,
    role: str,
    system_prompt: str,
    user_prompt: str,
    provider: str | None = None,
    environ: Mapping[str, str] | None = None,
    client_factory: ClientFactory | None = None,
    transport: Transport | None = None,
    diagnostics: MutableMapping[str, Any] | None = None,
) -> str:
    """Return provider output as plain text; JSON object validation stays in agents."""
    config = resolve_provider_config(role, provider=provider, environ=environ)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(config.diagnostics())
    try:
        if transport is not None:
            response = transport(config, system_prompt, user_prompt)
        else:
            factory = client_factory or _default_client_factory
            client = _get_client(config, factory)
            if config.provider == "gemini":
                response = client.models.generate_content(
                    model=config.model,
                    config={
                        "system_instruction": system_prompt,
                        "response_mime_type": "application/json",
                    },
                    contents=user_prompt,
                )
            else:
                response = client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
        return _extract_text(config, response)
    except (ProviderConfigurationError, ProviderRequestError):
        raise
    except Exception as exc:
        raise ProviderRequestError(
            f"{config.role} {config.provider} provider request failed: {type(exc).__name__}"
        ) from None
