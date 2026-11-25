#!/usr/bin/env python3
"""Download CIC IoT-DIAD 2024 while preserving its directory tree."""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import shutil
import sys
import time
import tomllib
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from threading import Lock
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


@dataclass
class Transfer:
    name: str
    status: str
    completed: int = 0
    total: int | None = None
    speed: float = 0.0
    updated_at: float = 0.0


class DownloadDashboard:
    def __init__(self, file_count: int, workers: int) -> None:
        self.file_count = file_count
        self.workers = workers
        self.started_at = time.monotonic()
        self.lock = Lock()
        self.active: dict[int, Transfer] = {}
        self.counts: Counter[str] = Counter()
        self.events: deque[tuple[str, str]] = deque(maxlen=5)
        self.color = "NO_COLOR" not in os.environ

    def __enter__(self):
        sys.stdout.write("\033[?1049h\033[?25l")
        sys.stdout.flush()
        return self

    def __exit__(self, *_exc):
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()

    def start(self, index: int, name: str) -> None:
        with self.lock:
            self.active[index] = Transfer(name, "CONNECT")

    def progress(self, index: int, completed: int, total: int | None, speed: float) -> None:
        with self.lock:
            transfer = self.active[index]
            transfer.status = "TRANSFER"
            transfer.completed = completed
            transfer.total = total
            transfer.speed = speed
            transfer.updated_at = time.monotonic()

    def retry(self, index: int, attempt: int, delay: int, error: Exception) -> None:
        with self.lock:
            transfer = self.active[index]
            transfer.status = f"RETRY {attempt}/3"
            transfer.speed = 0
            self.events.append(
                ("RETRY", f"{index:>3} {Path(transfer.name).name} in {delay}s | {error}")
            )

    def finish(self, index: int, status: str, error: str = "") -> None:
        with self.lock:
            transfer = self.active.pop(index)
            self.counts[status] += 1
            label = {"downloaded": "DONE", "skipped": "SKIP", "failed": "FAIL"}[status]
            detail = error or (format_bytes(transfer.completed) if status == "downloaded" else "")
            item = f"{index:>3} {Path(transfer.name).name}"
            self.events.append((label, f"{item} | {detail}" if detail else item))

    def draw(self) -> None:
        width, height = shutil.get_terminal_size((100, 24))
        with self.lock:
            active = [(index, replace(item)) for index, item in sorted(self.active.items())]
            counts = self.counts.copy()
            events = list(self.events)
        processed = counts.total()
        now = time.monotonic()
        speed = sum(item.speed for _, item in active if now - item.updated_at <= 3)
        pending = self.file_count - processed - len(active)
        bar_width = min(28, max(8, width // 4))
        filled = bar_width * processed // self.file_count
        lines = [
            ("1;36", "CIC IoT-DIAD 2024  /  DOWNLOAD MONITOR"),
            ("2", "-" * min(width - 1, 96)),
            (
                "36",
                f"[{'#' * filled}{'.' * (bar_width - filled)}]  "
                f"{processed}/{self.file_count} processed  |  "
                f"{format_duration(now - self.started_at)}",
            ),
            (
                "0",
                f"{counts['downloaded']} done  |  {counts['skipped']} skipped  |  "
                f"{counts['failed']} failed  |  {pending} queued",
            ),
            ("0", f"{len(active)}/{self.workers} active  |  ~{format_bytes(speed)}/s"),
        ]
        if active and height > 7:
            lines.extend([("0", ""), ("1;36", "ACTIVE TRANSFERS")])
            row_limit = max(1, height - len(lines) - (4 if events else 1))
            visible = active[:row_limit]
            for index, item in visible:
                speed_text = (
                    f" {format_bytes(item.speed)}/s"
                    if item.speed and now - item.updated_at <= 3
                    else ""
                )
                amount = format_bytes(item.completed) if item.completed or item.total else ""
                if item.total:
                    amount += f" {min(100, item.completed * 100 // item.total):>3}%"
                lines.append(
                    (
                        "33" if item.status.startswith("RETRY") else "0",
                        f"{index:>3}  {item.status:<9} {amount:<14}{speed_text:<14} "
                        f"{Path(item.name).name}",
                    )
                )
            if len(visible) < len(active):
                lines.append(("2", f"     +{len(active) - len(visible)} more active"))
        if events:
            lines.extend([("0", ""), ("1;36", "RECENT EVENTS")])
            room = max(0, height - len(lines) - 1)
            for label, message in events[-room:] if room else []:
                tone = {"DONE": "32", "SKIP": "2", "FAIL": "31", "RETRY": "33"}[label]
                lines.append((tone, f"{label:<5} {message}"))

        display = []
        for tone, line in lines[: max(1, height - 1)]:
            clean = "".join(char if char.isascii() and char.isprintable() else "?" for char in line)
            clipped = clean if len(clean) < width else clean[: max(0, width - 4)] + "..."
            display.append(f"\033[{tone}m{clipped}\033[0m" if self.color else clipped)
        sys.stdout.write("\033[H" + "\033[K\n".join(display) + "\033[J")
        sys.stdout.flush()


def download_one(
    opener,
    url: str,
    destination: Path,
    overwrite: bool,
    file_index: int,
    file_count: int,
    display_name: str,
    on_progress: Callable[[int, int | None, float], None] | None = None,
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
                if on_progress is not None and now - last_update >= PROGRESS_INTERVAL_SECONDS:
                    elapsed = max(now - started_at, 1e-9)
                    on_progress(starting_size + received, total_size, received / elapsed)
                    last_update = now

        elapsed = max(time.monotonic() - started_at, 1e-9)
        if response_size is not None and received != response_size:
            raise OSError(f"incomplete response: received {received} of {response_size} bytes")

    partial.replace(destination)
    if on_progress is not None:
        on_progress(starting_size + received, total_size, received / elapsed)
    else:
        print(
            f"[{file_index}/{file_count}] done {display_name} "
            f"({format_bytes(starting_size + received)}, {format_bytes(received / elapsed)}/s)",
            flush=True,
        )
    return "downloaded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Local defaults: $XDG_CONFIG_HOME/iot-diad/download.toml "
        "(normally ~/.config/iot-diad/download.toml). "
        "Use first_name, last_name, email, institution, job_title, country, and workers.",
    )
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
    parser.add_argument(
        "--workers",
        type=int,
        help="files downloaded in parallel (default: 4, or the local config value)",
    )
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    parser.add_argument("--plain", action="store_true", help="use plain logs in a terminal")
    args = parser.parse_args()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    config_path = config_home / "iot-diad" / "download.toml"
    try:
        with config_path.open("rb") as config_file:
            defaults = tomllib.load(config_file)
    except FileNotFoundError:
        defaults = {}
    for field in ("first_name", "last_name", "email", "institution", "job_title", "country"):
        if getattr(args, field) is None and field in defaults:
            value = defaults[field]
            if not isinstance(value, str) or not value.strip():
                parser.error(f"{config_path}: {field} must be a non-empty string")
            setattr(args, field, value)
    if args.workers is None:
        workers = defaults.get("workers", 4)
        if type(workers) is not int or workers < 1:
            parser.error(f"{config_path}: workers must be a positive integer")
        args.workers = workers
    return args


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

    workers = max(1, args.workers)
    use_dashboard = sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and not args.plain
    dashboard = DownloadDashboard(len(files), workers) if use_dashboard else None

    def fetch(item: tuple[int, tuple[str, str]]) -> tuple[str, str, str]:
        index, (relative_name, url) = item
        destination = safe_destination(args.output, relative_name)

        def update(completed: int, total: int | None, speed: float) -> None:
            assert dashboard is not None
            dashboard.progress(index, completed, total, speed)

        if dashboard:
            dashboard.start(index, relative_name)
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
                    update if dashboard else None,
                )
                if dashboard:
                    dashboard.finish(index, status)
                elif status == "skipped":
                    print(f"[{index}/{len(files)}] SKIP {relative_name}", flush=True)
                return status, relative_name, ""
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                if attempt == 3:
                    if dashboard:
                        dashboard.finish(index, "failed", str(exc))
                    else:
                        print(f"  FAILED {relative_name}: {exc}", file=sys.stderr, flush=True)
                    return "failed", relative_name, str(exc)
                delay = 2**attempt
                if dashboard:
                    dashboard.retry(index, attempt, delay, exc)
                else:
                    print(f"  retry {attempt}/3 in {delay}s ({relative_name}): {exc}", flush=True)
                time.sleep(delay)
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=workers) as pool, dashboard or nullcontext():
        if dashboard:
            dashboard.draw()
        futures = [pool.submit(fetch, item) for item in enumerate(files, start=1)]
        try:
            pending = set(futures)
            while dashboard and pending:
                _, pending = wait(pending, timeout=PROGRESS_INTERVAL_SECONDS)
                dashboard.draw()
            results = [future.result() for future in futures]
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
    downloaded = sum(status == "downloaded" for status, _, _ in results)
    skipped = sum(status == "skipped" for status, _, _ in results)
    failures = [(name, error) for status, name, error in results if status == "failed"]

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
