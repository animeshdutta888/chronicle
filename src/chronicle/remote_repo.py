from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess


def resolve_repo_path(
    repo: str | Path | None,
    repo_url: str | None,
    repos_dir: str | Path | None,
    branch: str | None = None,
) -> Path:
    """Return a local repository path, cloning/pulling when repo_url is provided."""
    if repo_url:
        base_dir = Path(repos_dir) if repos_dir else Path.cwd() / ".chronicle" / "repos"
        target = _target_path(base_dir, repo_url)
        if not target.exists():
            _run_git(["clone", repo_url, str(target)], cwd=base_dir.parent)
        else:
            _update_repo(target, branch=branch)
        if branch:
            _run_git(["-C", str(target), "checkout", branch], cwd=target)
            _run_git(["-C", str(target), "pull", "--ff-only", "origin", branch], cwd=target)
        return target.resolve()

    if repo is None:
        return Path(".").resolve()
    return Path(repo).resolve()


def _update_repo(repo_path: Path, branch: str | None = None) -> None:
    if not (repo_path / ".git").exists():
        raise RuntimeError(f"Target path exists but is not a git repo: {repo_path}")
    _run_git(["-C", str(repo_path), "fetch", "--all", "--prune"], cwd=repo_path)
    if branch:
        _run_git(["-C", str(repo_path), "checkout", branch], cwd=repo_path)
        _run_git(["-C", str(repo_path), "pull", "--ff-only", "origin", branch], cwd=repo_path)
        return
    pull = _run_git_optional(["-C", str(repo_path), "pull", "--ff-only"], cwd=repo_path)
    if pull.returncode != 0:
        stderr = (pull.stderr or "").lower()
        tolerated = [
            "no such ref was fetched",
            "there is no tracking information for the current branch",
            "your configuration specifies to merge with the ref",
        ]
        if not any(fragment in stderr for fragment in tolerated):
            message = pull.stderr.strip() or pull.stdout.strip() or "git pull failed"
            raise RuntimeError(message)


def _target_path(base_dir: Path, repo_url: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    slug = repo_url.rstrip("/").split("/")[-1]
    slug = slug[:-4] if slug.endswith(".git") else slug
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("-") or "repo"
    suffix = hashlib.sha1(repo_url.encode("utf-8")).hexdigest()[:8]
    return base_dir / f"{slug}-{suffix}"


def _run_git(args: list[str], cwd: Path) -> None:
    result = _run_git_optional(args=args, cwd=cwd)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise RuntimeError(message)


def _run_git_optional(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return result
