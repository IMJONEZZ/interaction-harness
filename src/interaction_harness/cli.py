"""interaction-harness command line.

    interaction-harness serve      the dashboard on http://localhost:8765
    interaction-harness doctor     check all four model roles and report each one's latency
    interaction-harness selftest   talk to the agent with synthesized speech and check every behaviour
    interaction-harness warmup     download and load the in-process speech models, then exit

Every subcommand takes the same settings flags; anything not given comes from the environment, .env, or
interaction-harness.toml (see .env.example). API keys come from the environment or files, never from flags,
so they stay out of shell history.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
import webbrowser

from . import __version__
from .config import LOCAL_BACKENDS, THINKING_MODES, Settings, load_settings

ROLES = ("foreground", "background", "asr", "tts")


def _add_settings_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", metavar="FILE", help="TOML settings (default: ./interaction-harness.toml if present)")
    g = p.add_argument_group("model roles", "each role is any OpenAI-compatible endpoint; ASR/TTS run "
                                            "in-process when their base URL is blank or 'local'")
    for role in ROLES:
        g.add_argument(f"--{role}-base-url", metavar="URL")
        g.add_argument(f"--{role}-model", metavar="NAME")
    g.add_argument("--foreground-thinking", choices=THINKING_MODES, help="how to switch thinking on that server")
    g.add_argument("--background-thinking", choices=THINKING_MODES)
    s = p.add_argument_group("speech")
    s.add_argument("--tts-voice", metavar="NAME")
    s.add_argument("--backend", choices=LOCAL_BACKENDS, help="in-process speech (default auto: mlx on Apple "
                                                             "Silicon, onnx elsewhere)")
    s.add_argument("--device", choices=("cpu", "cuda"), help="onnx backend device (default cpu)")
    p.add_argument("--tools", action="append", metavar="FILE", help="Python file of @tool functions (repeatable)")
    p.add_argument("--host", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default 8765)")


def _flags(args: argparse.Namespace) -> dict:
    keys = [f"{r}_{f}" for r in ROLES for f in ("base_url", "model")] + [
        "foreground_thinking", "background_thinking", "tts_voice", "backend", "device", "tools", "host", "port",
        "ssl_certfile", "ssl_keyfile"]
    return {k.upper(): getattr(args, k, None) for k in keys}


def _settings(args: argparse.Namespace, require_foreground: bool = True) -> Settings:
    try:
        return load_settings(_flags(args), config_path=args.config, require_foreground=require_foreground)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"interaction-harness: {e}")


def _local_url(settings: Settings) -> str:
    host = "localhost" if settings.host in ("127.0.0.1", "0.0.0.0", "::", "localhost") else settings.host
    return f"{'https' if settings.ssl_certfile else 'http'}://{host}:{settings.port}"


def cmd_serve(args: argparse.Namespace) -> None:
    import httpx
    import uvicorn

    from .server import create_app
    settings = _settings(args)
    url = _local_url(settings)
    print(f"interaction-harness {__version__}  ->  {url}", flush=True)
    if settings.host not in ("127.0.0.1", "localhost", "::1") and not settings.ssl_certfile:
        print("note: browsers only allow the microphone on localhost or HTTPS. From another machine, use an SSH "
              "tunnel (ssh -L 8765:localhost:8765 this-host) or pass --ssl-certfile/--ssl-keyfile.", flush=True)

    if args.open:
        def open_when_ready():
            for _ in range(600):
                try:
                    if httpx.get(f"{url}/api/status", timeout=1, verify=False).status_code == 200:
                        webbrowser.open(url)
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(1)
        threading.Thread(target=open_when_ready, daemon=True).start()

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning",
                ssl_certfile=settings.ssl_certfile, ssl_keyfile=settings.ssl_keyfile)


def cmd_doctor(args: argparse.Namespace) -> None:
    from .doctor import run
    sys.exit(asyncio.run(run(_settings(args))))


def cmd_selftest(args: argparse.Namespace) -> None:
    from .selftest import run
    sys.exit(asyncio.run(run(_settings(args), url=args.url)))


def cmd_warmup(args: argparse.Namespace) -> None:
    from .doctor import warmup
    sys.exit(asyncio.run(warmup(_settings(args, require_foreground=False))))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="interaction-harness", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the dashboard and the voice session server")
    _add_settings_flags(p)
    p.add_argument("--ssl-certfile", metavar="FILE", help="serve HTTPS so a remote browser may use the mic")
    p.add_argument("--ssl-keyfile", metavar="FILE")
    p.add_argument("--open", action="store_true", help="open the dashboard in a browser once it is ready")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("doctor", help="check every model role and report latency")
    _add_settings_flags(p)
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("selftest", help="end-to-end test with synthesized speech (starts a server if none is up)")
    _add_settings_flags(p)
    p.add_argument("--url", help="WebSocket of a running server (default: ws://127.0.0.1:<port>/ws, started "
                                 "in-process if nothing is listening)")
    p.set_defaults(fn=cmd_selftest)

    p = sub.add_parser("warmup", help="download and load the in-process speech models, then exit")
    _add_settings_flags(p)
    p.set_defaults(fn=cmd_warmup)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
