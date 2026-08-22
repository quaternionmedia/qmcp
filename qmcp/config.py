"""Configuration management for QMCP server."""

from functools import lru_cache

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

    # THE HARNESS ANSWERS ON PI. Three services on one machine need three ports
    # somebody can recall without looking, and the constants do that better
    # than a block of neighbouring numbers: 3141 the harness, 1618 the panel,
    # 2718 the maps. `1337` was the obvious joke and is already in use on this
    # machine, which is the argument against a port everybody thinks of.
    #
    # Not 8000 and not 3333: 8000 is codecarto's old default and half the
    # Python world's, and 3333 is what this served while the panel looked on
    # 8000 -- a mismatch that had the panel reporting an absent archive while
    # this served 203 threads.
    #
    # Override with `QMCP_PORT`, or `--port` on the command line. The env
    # prefix above makes every setting here overridable the same way.
    port: int = 3141
    debug: bool = False

    # Database settings (Phase 2)
    database_url: str = "sqlite+aiosqlite:///./qmcp.db"

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
