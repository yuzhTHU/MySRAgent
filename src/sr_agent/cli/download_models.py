# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Download sr-agent model checkpoints."""
from __future__ import annotations
import argparse
import dotenv
from sr_agent.utils import tag2ansi, download_model, get_default

dotenv.load_dotenv()


def build_common_parser(description: str) -> argparse.ArgumentParser:
    default_repo = get_default("repo")
    default_release_tag = get_default("release_tag")
    default_token = get_default("token")

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint path.")
    parser.add_argument("--name", required=True, help="Remote model name, e.g. property-scratch.")
    parser.add_argument("--repo", default=default_repo, help=f"GitHub repo in owner/name form.")
    parser.add_argument("--release-tag", default=default_release_tag, help=f"GitHub release tag.")
    parser.add_argument("--token", default=default_token, help="GitHub token.")
    return parser


def main() -> None:
    parser = build_common_parser("Download a model checkpoint from the sr-agent model store.")
    args = parser.parse_args()
    save_path = download_model(
        name=args.name,
        checkpoint=args.checkpoint,
        repo=args.repo,
        release_tag=args.release_tag,
        token=args.token,
    )
    print(tag2ansi(f"Downloaded [bold green]{args.name}[reset] to [bold blue]{save_path}[reset]"))


if __name__ == "__main__":
    main()
