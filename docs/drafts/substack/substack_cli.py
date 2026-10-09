"""Run the Substack CLI using the publication domain for authentication."""

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from substack import Api, cli


def publication_api(**kwargs: Any) -> Api:
    """Keep the upstream CLI behavior while changing its authentication origin."""
    publication_url = kwargs.get("publication_url")
    if not publication_url:
        raise ValueError("Set PUBLICATION_URL in the local .env file.")
    parsed = urlparse(publication_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path.strip("/"):
        raise ValueError("PUBLICATION_URL must be an HTTPS publication URL.")
    kwargs["base_url"] = publication_url.rstrip("/") + "/api/v1"
    return Api(**kwargs)


def main(argv: list[str] | None = None) -> int:
    """Load local settings and delegate arguments, errors, and prompts to the CLI."""
    load_dotenv(Path(__file__).with_name(".env"))
    # Settings were loaded explicitly; prevent the CLI from finding a parent .env.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    original_api = cli.Api
    cli.Api = publication_api
    try:
        return cli.main(argv)
    finally:
        cli.Api = original_api


if __name__ == "__main__":
    raise SystemExit(main())
