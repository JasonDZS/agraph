"""AGraph API Server CLI."""

import argparse
import os
import sys
from typing import Any, Dict, Optional

from ..logger import logger


def main() -> None:
    """Main entry point for AGraph API server."""
    parser = argparse.ArgumentParser(
        description="AGraph API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  agraph-server                          # Start server with default settings
  agraph-server --host 0.0.0.0 --port 8080
  agraph-server --reload                 # Development mode with auto-reload
  agraph-server --workers 4             # Production with multiple workers
        """,
    )

    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind the server to (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to (default: 8000)"
    )

    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (production only)"
    )

    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
        help="Log level (default: info)",
    )

    parser.add_argument("--access-log", action="store_true", help="Enable access log")

    parser.add_argument("--env-file", help="Path to .env file")

    args = parser.parse_args()

    # Check if FastAPI dependencies are available
    try:
        import uvicorn  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.error("FastAPI dependencies not installed. Install with: uv add agraph[api]")
        sys.exit(1)

    # Set environment file if specified
    if args.env_file:
        if os.path.exists(args.env_file):
            os.environ["AGRAPH_ENV_FILE"] = args.env_file
        else:
            logger.warning(f"Environment file not found: {args.env_file}")

    logger.info("Starting AGraph API Server...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Log level: {args.log_level}")

    # Configure uvicorn settings
    uvicorn_config = {
        "app": "agraph.api.app:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "access_log": args.access_log,
    }

    if args.reload:
        # Development mode
        uvicorn_config.update(
            {
                "reload": True,
                "reload_dirs": ["agraph"],
            }
        )
        logger.info("Running in development mode with auto-reload")
    elif args.workers > 1:
        # Production mode with multiple workers
        uvicorn_config["workers"] = args.workers
        logger.info(f"Running in production mode with {args.workers} workers")

    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


def run_with_gunicorn(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 4,
    worker_class: str = "uvicorn.workers.UvicornWorker",
    access_log: Optional[str] = None,
    error_log: Optional[str] = None,
) -> None:
    """Run server with Gunicorn for production deployment."""
    try:
        import gunicorn.app.base  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.error("Gunicorn not installed. Install with: uv add agraph[server]")
        sys.exit(1)

    class StandaloneApplication(gunicorn.app.base.BaseApplication):
        def __init__(self, app: Any, options: Optional[Dict[str, Any]] = None) -> None:
            self.options = options or {}
            self.application = app
            super().__init__()

        def init(self, parser: Any, opts: Any, args: Any) -> None:
            """Initialize the application."""
            return

        def load_config(self) -> None:
            for key, value in self.options.items():
                if key in self.cfg.settings and value is not None:
                    self.cfg.set(key.lower(), value)

        def load(self) -> Any:
            return self.application

    options = {
        "bind": f"{host}:{port}",
        "workers": workers,
        "worker_class": worker_class,
        "worker_connections": 1000,
        "max_requests": 1000,
        "max_requests_jitter": 100,
        "timeout": 30,
        "keepalive": 2,
        "preload_app": True,
    }

    if access_log:
        options["accesslog"] = access_log
    if error_log:
        options["errorlog"] = error_log

    from .app import app  # pylint: disable=import-outside-toplevel

    logger.info(f"Starting AGraph API Server with Gunicorn on {host}:{port}")
    logger.info(f"Workers: {workers}, Worker class: {worker_class}")

    StandaloneApplication(app, options).run()


if __name__ == "__main__":
    main()
