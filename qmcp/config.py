"""Configuration management for QMCP server."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="QMCP_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server settings
    host: str = "127.0.0.1"

    # The harness answers on pi. Three services on one machine take ports
    # named for constants, recalled without looking: 3141 the harness, 1618
    # the panel, 2718 the maps. 8000, the common Python default, is avoided.
    #
    # Override with `QMCP_PORT`, or `--port` on the command line. The env
    # prefix above makes every setting here overridable the same way.
    port: int = 3141
    debug: bool = False

    # Database settings (Phase 2)
    database_url: str = "sqlite+aiosqlite:///./qmcp.db"

    # Logging
    log_level: str = "INFO"

    # Which vox engine adapter answers `POST /v1/human/requests/{id}/voice`,
    # and where it listens when not at the adapter's own default.
    voice_engine: str = "joe"
    voice_engine_url: str | None = None

    # The spoken conversation `qmcp serve --converse` starts beside the server:
    # the runtime it acts with (None: no conversation), where a project acted on
    # for the first time is looked for, and a word an instruction must begin with.
    converse_runtime: str | None = None
    converse_clones: str | None = None
    converse_wake: str | None = None
    converse_synth: str = "pyttsx3"
    # How sure the speech engine must be of a spoken instruction for its
    # read-back to ask nothing, silence agreeing; above 1 always asks.
    voice_tacit_confidence: float = 0.7
    # Whether an instruction's take is hinted with the core projects' terms as
    # well as their names (`qmcp vocabulary`); off until real takes say it helps.
    voice_hint_terms: bool = False

    # The alfred the `alfred_*` tools drive (docs/integrations/alfred.md): where
    # its API answers, and who asks -- an account qmcp logs in as, or a token
    # alfred issued, or neither for an alfred that asks for no account. With no
    # URL the tools answer that alfred is not configured. Seconds alfred has to
    # answer one request, and to answer a delivery, which it does only once it
    # has copied or uploaded the whole render.
    alfred_url: str | None = None
    alfred_email: str | None = None
    alfred_password: SecretStr | None = None
    alfred_token: SecretStr | None = None
    alfred_timeout: float = 10.0
    alfred_deliver_timeout: float = 300.0


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
