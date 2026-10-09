"""Run the Substack CLI using the publication domain for authentication."""

from html.parser import HTMLParser
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from substack import Api, cli
import yaml


class PlainText(HTMLParser):
    """Extract display text from Pandoc metadata containing HTML markup."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def metadata_text(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Front matter {key} must be a string.")
    parser = PlainText()
    parser.feed(value)
    return "".join(parser.parts).strip()


def split_front_matter(markdown: str) -> tuple[dict[str, Any], str]:
    """Remove only a leading YAML metadata block; preserve the remaining body."""
    lines = markdown.splitlines(keepends=True)
    if not lines or lines[0].lstrip("\ufeff").strip() != "---":
        return {}, markdown
    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            try:
                metadata = yaml.safe_load("".join(lines[1:index]))
            except yaml.YAMLError as exc:
                raise ValueError("Invalid YAML front matter.") from exc
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError("YAML front matter must contain a mapping.")
            return metadata, "".join(lines[index + 1 :])
    # A leading horizontal rule without a closing metadata delimiter is Markdown.
    return {}, markdown


class DispatchApi(Api):
    """Strip local metadata before either create or update sends the body."""

    def create_draft_from_markdown(
        self,
        title: str,
        markdown: str,
        subtitle: str = "",
        audience: str = "everyone",
        write_comment_permissions: str = "everyone",
        search_engine_title: str = None,
        search_engine_description: str = None,
        slug: str = None,
        draft_section_id: int = None,
        tags=None,
        prepublish: bool = False,
        publish: bool = False,
        send: bool = True,
        share_automatically: bool = False,
    ) -> dict:
        _, body = split_front_matter(markdown)
        return super().create_draft_from_markdown(
            title=title,
            markdown=body,
            subtitle=subtitle,
            audience=audience,
            write_comment_permissions=write_comment_permissions,
            search_engine_title=search_engine_title,
            search_engine_description=search_engine_description,
            slug=slug,
            draft_section_id=draft_section_id,
            tags=tags,
            prepublish=prepublish,
            publish=publish,
            send=send,
            share_automatically=share_automatically,
        )

    def update_draft_from_markdown(
        self,
        draft_id: int,
        markdown: str,
        *,
        subtitle: str = None,
        audience: str = None,
        write_comment_permissions: str = None,
        search_engine_title: str = None,
        search_engine_description: str = None,
        slug: str = None,
        draft_section_id: int = None,
        tags=None,
        dry_run: bool = False,
        allow_unsupported_change: bool = False,
        allow_image_replacement: bool = False,
        expected_revision=None,
        allow_conflict: bool = False,
        force: bool = False,
    ) -> dict:
        _, body = split_front_matter(markdown)
        return super().update_draft_from_markdown(
            draft_id=draft_id,
            markdown=body,
            subtitle=subtitle,
            audience=audience,
            write_comment_permissions=write_comment_permissions,
            search_engine_title=search_engine_title,
            search_engine_description=search_engine_description,
            slug=slug,
            draft_section_id=draft_section_id,
            tags=tags,
            dry_run=dry_run,
            allow_unsupported_change=allow_unsupported_change,
            allow_image_replacement=allow_image_replacement,
            expected_revision=expected_revision,
            allow_conflict=allow_conflict,
            force=force,
        )


def metadata_arguments(arguments: list[str]) -> list[str]:
    """Use front-matter defaults while preserving explicit CLI metadata options."""
    args = cli._build_parser().parse_args(arguments)
    if args.command != "drafts" or args.draft_command not in {"create", "update"}:
        return arguments
    metadata, _ = split_front_matter(
        Path(args.markdown_file).read_text(encoding="utf-8")
    )
    result = list(arguments)
    keys = ["subtitle"]
    if args.draft_command == "create":
        keys.append("title")
    for key in keys:
        option = "--" + key
        explicit = any(a == option or a.startswith(option + "=") for a in arguments)
        if key in metadata and not explicit:
            result.extend([option, metadata_text(metadata[key], key)])
    return result


def publication_api(**kwargs: Any) -> Api:
    """Keep the upstream CLI behavior while changing its authentication origin."""
    publication_url = kwargs.get("publication_url")
    if not publication_url:
        raise ValueError("Set PUBLICATION_URL in the local .env file.")
    parsed = urlparse(publication_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path.strip("/"):
        raise ValueError("PUBLICATION_URL must be an HTTPS publication URL.")
    kwargs["base_url"] = publication_url.rstrip("/") + "/api/v1"
    return DispatchApi(**kwargs)


def main(argv: list[str] | None = None) -> int:
    """Load local settings and delegate arguments, errors, and prompts to the CLI."""
    load_dotenv(Path(__file__).with_name(".env"))
    # Settings were loaded explicitly; prevent the CLI from finding a parent .env.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    original_api = cli.Api
    cli.Api = publication_api
    try:
        arguments = list(argv if argv is not None else sys.argv[1:])
        try:
            arguments = metadata_arguments(arguments)
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return cli.main(arguments)
    finally:
        cli.Api = original_api


if __name__ == "__main__":
    raise SystemExit(main())
