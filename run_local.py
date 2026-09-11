"""Run the local OCR API with the repository virtual environment."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from auth import create_admin_session_token
from config import (
    ADMIN_EDIT_PASSWORD,
    BASE_DIR,
    DEFAULT_STREAM_URL,
    SAMPLE_INTERVAL_SECONDS,
    SESSION_COOKIE_NAME,
    SESSION_SECRET,
)

WORKSPACE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = WORKSPACE_DIR
VENV_PYTHON = WORKSPACE_DIR / ".venv" / "bin" / "python"
TWITCH_STREAM_URL = DEFAULT_STREAM_URL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local OCR API.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the API to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the API to.")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable automatic reload when source files change.",
    )
    parser.add_argument(
        "--debug-ocr",
        action="store_true",
        help="Write OCR threshold images under backend/test-assets.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Automatically start an OCR job on startup.",
    )
    parser.add_argument(
        "--stream-url",
        default=None,
        help="Custom stream URL to auto-start with (enables auto-start).",
    )
    return parser.parse_args()

def wait_for_api(base_url: str, process: subprocess.Popen[bytes]) -> bool:
    for _ in range(30):
        if process.poll() is not None:
            return False
        try:
            with urlopen(f"{base_url}/openapi.json", timeout=1):
                return True
        except URLError:
            time.sleep(0.5)
    return False


def create_admin_session(base_url: str) -> str:
    """Obtain or generate a valid admin session cookie for local execution."""
    if ADMIN_EDIT_PASSWORD:
        payload = json.dumps({"password": ADMIN_EDIT_PASSWORD}).encode("utf-8")
        request = Request(
            f"{base_url}/admin/session",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:
                session_cookie = response.headers.get("Set-Cookie")
                if session_cookie:
                    return session_cookie.split(";", maxsplit=1)[0]
        except Exception:
            pass

    # Fallback to direct HMAC token generation if API call failed
    return f"{SESSION_COOKIE_NAME}={create_admin_session_token()}"


def start_ocr_job(base_url: str, stream_url: str, session_cookie: str) -> None:
    payload = json.dumps(
        {
            "stream_url": stream_url,
            "seconds_interval": SAMPLE_INTERVAL_SECONDS,
            "realtime": True,
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url}/run-local-video",
        data=payload,
        headers={"Content-Type": "application/json", "Cookie": session_cookie},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        payload = json.loads(response)
        print(f"OCR job status: {payload['status']}")
    
def stop_ocr_api(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGINT)
        try:
            process.wait(timeout=3)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            print("OCR API did not stop after SIGINT; sending SIGTERM.")
            os.killpg(process_group, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                print("OCR API did not stop after SIGTERM; sending SIGKILL.")
                os.killpg(process_group, signal.SIGKILL)
                process.wait(timeout=3)
    except ProcessLookupError:
        pass
    except KeyboardInterrupt:
        pass

def main() -> int:
    args = parse_args()
    process: subprocess.Popen[bytes] | None = None

    if not VENV_PYTHON.is_file():
        print(f"Missing virtual environment interpreter: {VENV_PYTHON}", file=sys.stderr)
        return 1

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(BACKEND_DIR)
    environment["ADMIN_EDIT_PASSWORD"] = ADMIN_EDIT_PASSWORD
    environment["SESSION_SECRET"] = SESSION_SECRET
    if args.debug_ocr:
        environment["DEBUG_OCR"] = "true"

    command = [
        str(VENV_PYTHON),
        "-m",
        "uvicorn",
        "endpoints:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.no_reload:
        command.append("--reload")

    base_url = f"http://{args.host}:{args.port}"
    print("Starting local OCR API")
    print(f"Docs: {base_url}/docs")
    print(f"Live events: {base_url}/game-state-events")

    try:
        process = subprocess.Popen(
            command,
            cwd=BACKEND_DIR,
            env=environment,
            start_new_session=True,
        )
        if not wait_for_api(base_url, process):
            if process.poll() is None:
                print("OCR API did not start within 15 seconds.", file=sys.stderr)
            else:
                print("OCR API stopped before it became ready.", file=sys.stderr)
            return process.wait()

        auto_stream_url = args.stream_url or (TWITCH_STREAM_URL if args.auto_start else None)
        if auto_stream_url:
            try:
                start_ocr_job(base_url, auto_stream_url, create_admin_session(base_url))
            except URLError as error:
                print(f"Unable to start OCR: {error}", file=sys.stderr)
        else:
            print("OCR server is idle. Submit a stream URL from the frontend or API to start processing.")

        return process.wait()
    except KeyboardInterrupt:
        if process:
            stop_ocr_api(process)
        return 0
    except FileNotFoundError as error:
        print(f"Unable to start Uvicorn: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())