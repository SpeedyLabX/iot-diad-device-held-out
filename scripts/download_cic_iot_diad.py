#!/usr/bin/env python3
"""Download CIC IoT-DIAD 2024 while preserving its directory tree."""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import time
from collections import deque
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener

BASE_URL = "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/"
REGISTER_URL = urljoin(BASE_URL, "insert.php")
ROOT_BROWSE_URL = urljoin(BASE_URL, "browse.php?p=")
USER_AGENT = "CIC-IoT-DIAD-downloader/1.0"
PROGRESS_INTERVAL_SECONDS = 0.5


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def prompt(value: str | None, label: str, default: str | None = None) -> str:
    if value:
        return value
    suffix = f" [{default}]" if default else ""
    entered = input(f"{label}{suffix}: ").strip()
    result = entered or default
    if not result:
        raise SystemExit(f"Missing required value: {label}")
    return result


def make_opener():
    cookie_jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener


def register(opener, fields: dict[str, str]) -> None:
    request = Request(
        REGISTER_URL,
        data=urlencode(fields).encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": BASE_URL,
        },
        method="POST",
    )
    with opener.open(request, timeout=60) as response:
        raw = response.read().decode("utf-8", errors="replace")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unexpected registration response: {raw[:300]}") from exc
    if not result.get("ok"):
        raise RuntimeError(result.get("message") or "CIC registration failed")
    print(result.get("message") or "CIC registration succeeded.")


def read_text(opener, url: str) -> str:
    with opener.open(url, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    if "Registration required" in text:
        raise RuntimeError("CIC session is not registered or has expired")
    return text


def discover_files(opener) -> list[tuple[str, str]]:
    pages = deque([ROOT_BROWSE_URL])
    visited: set[str] = set()
    files: dict[str, str] = {}
    base = urlparse(BASE_URL)

    while pages:
        page_url = pages.popleft()
        if page_url in visited:
            continue
        visited.add(page_url)

        parser = LinkParser()
        parser.feed(read_text(opener, page_url))

        for href in parser.links:
            candidate = urljoin(page_url, href)
            parsed = urlparse(candidate)
            if parsed.netloc != base.netloc or not parsed.path.startswith(base.path):
                continue
            if parsed.path.endswith("/browse.php"):
                if candidate not in visited:
                    pages.append(candidate)
            elif parsed.path.endswith("/download.php"):
                values = parse_qs(parsed.query).get("file")
                if values and values[0]:
                    files[values[0]] = candidate

        print(
            f"Discovering: {len(visited)} pages, {len(files)} files",
            end="\r",
            flush=True,
        )

    print()
    return sorted(files.items())


def safe_destination(root: Path, relative_name: str) -> Path:
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe dataset path: {relative_name!r}")
    destination = root.joinpath(*relative.parts).resolve()
    root_resolved = root.resolve()
    if root_resolved not in destination.parents:
        raise ValueError(f"Dataset path escapes output directory: {relative_name!r}")
    return destination


def format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def show_progress(
    file_index: int,
    file_count: int,
    name: str,
    completed: int,
    total: int | None,
    speed: float,
) -> None:
    if total:
        percent = min(completed / total * 100, 100.0)
        remaining = max(total - completed, 0)
        eta = remaining / speed if speed > 0 else float("inf")
        amount = f"{format_bytes(completed)}/{format_bytes(total)} {percent:5.1f}%"
        eta_text = f"ETA {format_duration(eta)}"
    else:
        amount = format_bytes(completed)
        eta_text = "ETA --"
    line = f"[{file_index}/{file_count}] {name} | {amount} | {format_bytes(speed)}/s | {eta_text}"
    sys.stdout.write("\r\033[K" + line)
    sys.stdout.flush()


def download_one(
    opener,
    url: str,
    destination: Path,
    overwrite: bool,
    file_index: int,
    file_count: int,
    display_name: str,
) -> str:
    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = Request(url, headers={"Referer": ROOT_BROWSE_URL})
    if existing:
        request.add_header("Range", f"bytes={existing}-")

    with opener.open(request, timeout=120) as response:
        append = existing > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        starting_size = existing if append else 0
        content_length = response.headers.get("Content-Length")
        response_size = int(content_length) if content_length else None
        total_size = starting_size + response_size if response_size is not None else None
        received = 0
        started_at = time.monotonic()
        last_update = 0.0
        with partial.open(mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                now = time.monotonic()
                if now - last_update >= PROGRESS_INTERVAL_SECONDS:
                    elapsed = max(now - started_at, 1e-9)
                    show_progress(
                        file_index,
                        file_count,
                        display_name,
                        starting_size + received,
                        total_size,
                        received / elapsed,
                    )
                    last_update = now

        elapsed = max(time.monotonic() - started_at, 1e-9)
        show_progress(
            file_index,
            file_count,
            display_name,
            starting_size + received,
            total_size,
            received / elapsed,
        )
        sys.stdout.write("\n")
        if response_size is not None and received != response_size:
            raise OSError(f"incomplete response: received {received} of {response_size} bytes")

    partial.replace(destination)
    return "downloaded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/CIC-IoT-IDAD-Dataset-2024"),
        help="output directory (default: %(default)s)",
    )
    parser.add_argument("--first-name")
    parser.add_argument("--last-name")
    parser.add_argument("--email")
    parser.add_argument("--institution")
    parser.add_argument("--job-title")
    parser.add_argument("--country")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fields = {
        "first_name": prompt(args.first_name, "First name"),
        "last_name": prompt(args.last_name, "Last name"),
        "email": prompt(args.email, "Email"),
        "institution": prompt(args.institution, "Institution"),
        "job_title": prompt(args.job_title, "Job title"),
        "country": prompt(args.country, "Country"),
    }

    opener = make_opener()
    register(opener, fields)
    files = discover_files(opener)
    if not files:
        raise RuntimeError("No downloadable files were discovered")

    print(f"Found {len(files)} files.")
    print(f"Output: {args.output.resolve()}")
    if not args.yes:
        answer = input("Start downloading? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return 0

    downloaded = skipped = 0
    failures: list[tuple[str, str]] = []
    for index, (relative_name, url) in enumerate(files, start=1):
        destination = safe_destination(args.output, relative_name)
        for attempt in range(1, 4):
            try:
                status = download_one(
                    opener,
                    url,
                    destination,
                    args.overwrite,
                    index,
                    len(files),
                    relative_name,
                )
                if status == "skipped":
                    skipped += 1
                    print(f"[{index}/{len(files)}] SKIP {relative_name}")
                else:
                    downloaded += 1
                break
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                if attempt == 3:
                    failures.append((relative_name, str(exc)))
                    print(f"  FAILED: {exc}", file=sys.stderr)
                else:
                    delay = 2**attempt
                    print(f"  retry {attempt}/3 in {delay}s: {exc}")
                    time.sleep(delay)

    print(f"Complete: {downloaded} downloaded, {skipped} skipped, {len(failures)} failed")
    if failures:
        print("Failed files:", file=sys.stderr)
        for name, error in failures:
            print(f"- {name}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the command to continue partial downloads.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
