# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Utilities for uploading and downloading model checkpoints."""
from __future__ import annotations
import os
import shutil
import logging
import requests
from pathlib import Path
from typing import Literal, Optional
from .log_exception import log_exception


_logger = logging.getLogger(f"sr_agent.{__name__}")
__all__ = ["get_default", "download_model", "upload_model"]
DEFAULT_GITHUB_REPO = "yuzhTHU/MySRAgent"
DEFAULT_GITHUB_RELEASE_TAG = "sr-agent-models"
GITHUB_HEADER = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


class ModelStoreError(RuntimeError):
    """Raised when a model store operation fails."""


def get_default(key: Literal["repo", "release_tag", "token"]) -> Optional[str]:
    if key == "repo":
        return os.getenv("SR_AGENT_GITHUB_REPO", DEFAULT_GITHUB_REPO)
    elif key == "release_tag":
        return os.getenv("SR_AGENT_GITHUB_RELEASE_TAG", DEFAULT_GITHUB_RELEASE_TAG)
    elif key == "token":
        return os.getenv("GITHUB_TOKEN")
    else:
        raise ValueError(f"Invalid key: {key}")


def download_model(name: str, checkpoint: str, repo: str = None, release_tag: str = None, token: str = None) -> Path:
    """ 从 Github 的 {repo} 仓库的 {release_tag} 版本中下载名为 {name} 的模型文件，并保存到本地路径 {checkpoint}。 """
    if repo is None: repo = get_default("repo")
    if release_tag is None: release_tag = get_default("release_tag")
    if token is None: token = get_default("token")
    
    checkpoint_path = Path(checkpoint).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".downloading")
    
    try:
        download_url = get_download_url(repo, release_tag, name, token)
        headers = GITHUB_HEADER | {"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"}
        with requests.get(download_url, headers=headers, stream=True, timeout=120) as response:
            raise_for_status(response, "download model asset")
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk: f.write(chunk)
        shutil.move(str(tmp_path), checkpoint_path)
    except Exception as e:
        _logger.error(f"Failed to download model {name} from {repo}@{release_tag}: {log_exception(e, with_traceback=False)}")
        checkpoint_path = None
    finally:
        tmp_path.unlink(missing_ok=True)
    return checkpoint_path


def upload_model(name: str, checkpoint: str, repo: str = None, release_tag: str = None, token: str = None) -> dict:
    """ 从本地路径 {checkpoint} 上传模型文件到 Github 的 {repo} 仓库的 {release_tag} 版本，命名为 {name}。 """
    if repo is None: repo = get_default("repo")
    if release_tag is None: release_tag = get_default("release_tag")
    if token is None: token = get_default("token")

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if not token:
        raise ModelStoreError("GITHUB_TOKEN is required to upload model checkpoints to GitHub Releases.")
    
    try:
        upload_url = get_upload_url(repo, release_tag, name, token)
        headers = GITHUB_HEADER | {"Content-Type": "application/octet-stream", "Authorization": f"Bearer {token}"}
        with open(checkpoint_path, "rb") as f:
            with requests.post(upload_url, params={"name": name}, headers=headers, data=f, timeout=300) as response:
                raise_for_status(response, "upload model asset")
                return response.json()
    except Exception as e:
        raise ModelStoreError(f"Failed to upload model {name} to {repo}@{release_tag}: {e}") from e


def get_download_url(repo: str, release_tag: str, name: str, token: str = None) -> str:
    url = f"https://api.github.com/repos/{repo}/releases/tags/{release_tag}"
    headers = GITHUB_HEADER | {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, timeout=60) as response:
        raise_for_status(response, "get release for download")
        release = response.json() or {}
    for asset in release.get("assets", []):
        if asset.get("name") == name:
            return asset['url']
    else:
        raise ModelStoreError(f"Model asset not found in {repo}@{release_tag}: {name}")


def get_upload_url(repo: str, release_tag: str, name: str, token: str) -> str:
    release = None
    headers = GITHUB_HEADER | {"Authorization": f"Bearer {token}"}

    # 检索已有 Release
    if True:
        url = f"https://api.github.com/repos/{repo}/releases/tags/{release_tag}"
        with requests.get(url, headers=headers, timeout=60) as response:
            if response.status_code == 404:
                release = None
            else:
                raise_for_status(response, "get release for upload")
                release = response.json()

    # 若 Release 不存在则创建
    if release is None:
        url = f"https://api.github.com/repos/{repo}/releases"
        data = {
            "tag_name": release_tag, "name": release_tag,
            "body": "Model checkpoints used by sr-agent.",
            "draft": False, "prerelease": False,
        }
        with requests.post(url, headers=headers, json=data, timeout=60) as response:
            raise_for_status(response, "create GitHub release")
            release = response.json()

    # 删除已有或刚刚创建的 Release
    for asset in release.get("assets", []):
        if asset.get("name") == name:
            url = f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}"
            with requests.delete(url, headers=headers, timeout=60) as response:
                raise_for_status(response, "delete existing model asset")

    # 返回上传 URL
    return release["upload_url"].split("{", 1)[0]


def raise_for_status(response: requests.Response, action: str) -> None:
    if response.ok:
        return
    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    hint = ""
    if response.status_code == 403 and isinstance(detail, dict):
        message = str(detail.get("message", ""))
        if "Resource not accessible by personal access token" in message:
            hint = (
                " Hint: make sure the token is authorized for this repository, "
                "has Contents: Read and write permission, and has been approved/SSO-authorized "
                "by the organization if required."
            )
    raise ModelStoreError(f"Failed to {action}: HTTP {response.status_code}: {detail}.{hint}")
