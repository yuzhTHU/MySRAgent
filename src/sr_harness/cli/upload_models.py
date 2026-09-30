# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Upload SRHarness model checkpoints."""
from __future__ import annotations
import argparse
import dotenv
from sr_harness.utils import tag2ansi, upload_model, get_default

dotenv.load_dotenv()


def setup_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    default_repo = get_default("repo")
    default_release_tag = get_default("release_tag")
    default_token = get_default("token")

    description = "Upload a model checkpoint to the SRHarness model store."
    if parser is None:
        parser = argparse.ArgumentParser(prog="sr-harness upload-models", description=description)
    else:
        parser.description = description
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint path.")
    parser.add_argument("--name", required=True, help="Remote model name, e.g. property-scratch.")
    parser.add_argument("--repo", default=default_repo, help=f"GitHub repo in owner/name form.")
    parser.add_argument("--release-tag", default=default_release_tag, help=f"GitHub release tag.")
    parser.add_argument("--token", default=default_token, help="GitHub token.")
    return parser


def main(args: argparse.Namespace) -> int:
    asset = upload_model(
        name=args.name,
        checkpoint=args.checkpoint,
        repo=args.repo,
        release_tag=args.release_tag,
        token=args.token,
    )
    print(tag2ansi(f"Uploaded [bold green]{args.name}[reset]: [bold blue]{asset.get('browser_download_url', asset.get('url'))}[reset]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(setup_parser().parse_args()))
