"""Project credentials and the shared image for private Modal workspaces."""

import os
from functools import cache

from modal import Image
from pydantic import Field, SecretStr
from pydantic_ai_harness import modal_sandbox
from pydantic_settings import BaseSettings, SettingsConfigDict

from mam_bench.kludge import modal_filesystem_error_workaround


class ModalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODAL_",
        env_file=".env",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    token_id: SecretStr = Field(min_length=1)
    token_secret: SecretStr = Field(min_length=1)


@cache
def sandbox_image() -> Image:
    """Define once; Modal builds and caches the same image for all identities."""
    return (
        Image.debian_slim(python_version="3.14")
        .apt_install("git", "ripgrep", "util-linux")
        .pip_install("numpy==2.5.2")
        .workdir("/workspace")
    )


def new_workspace(timeout_seconds: int) -> modal_sandbox.ModalSandboxBackend:
    credentials = ModalSettings()  # pyright: ignore[reportCallIssue]
    # The SDK reads host credentials from the environment. Do not pass them as sandbox env.
    os.environ["MODAL_TOKEN_ID"] = credentials.token_id.get_secret_value()
    os.environ["MODAL_TOKEN_SECRET"] = credentials.token_secret.get_secret_value()
    # Pin this project's image builder; older account defaults reject Python 3.14.
    os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2025.06"
    modal_filesystem_error_workaround()
    return modal_sandbox.ModalSandboxBackend(
        image=sandbox_image(),
        app_name="mam-bench",
        working_dir="/workspace",
        sandbox_timeout=timeout_seconds,
    )
