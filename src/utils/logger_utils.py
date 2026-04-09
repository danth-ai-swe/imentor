import atexit
import contextlib
import functools
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from src.config.app_config import get_app_config

settings = get_app_config()

START_TIME = datetime.now()
BASE_DIR = Path(__file__).parent.parent.parent

_TIME_FMT = "%Y-%m-%d %H:%M:%S"


class AlignedLocationFormatter(logging.Formatter):
    def format(self, record):
        record.process_id = os.getpid()
        record.relative_created_seconds = record.relativeCreated / 1000.0

        if hasattr(record, 'pathname') and record.pathname:
            try:
                rel_path = Path(record.pathname).relative_to(BASE_DIR)
                location = f"{rel_path}:{record.lineno}"
            except ValueError:
                location = f"{record.pathname}:{record.lineno}"
        else:
            location = "unknown:0"

        record.location = location.ljust(50)[:50]
        return super().format(record)


class ColorFormatter(AlignedLocationFormatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m\033[97m",
    }
    RESET = "\033[0m"

    def format(self, record):
        super().format(record)
        log_color = self.COLORS.get(record.levelno, self.RESET)
        message = super(AlignedLocationFormatter, self).format(record)
        return f"{log_color}{message}{self.RESET}"


def get_system_info():
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "processor": platform.processor(),
        "pid": os.getpid(),
    }


class CustomLogger:
    _instance: "CustomLogger | None" = None

    def __new__(cls, app_name: str = "APP") -> "CustomLogger":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._init_logger(app_name)
            cls._instance = inst
        return cls._instance

    def _init_logger(self, app_name: str) -> None:
        self._logger = logging.getLogger(app_name)
        self._logger.setLevel(logging.DEBUG)

        console_format = (
            "[%(asctime)s]"
            "[%(levelname)-8s]"
            "[%(location)-50s]"
            " %(message)s"
        )
        date_format = _TIME_FMT

        if not self._logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = ColorFormatter(console_format, datefmt=date_format)
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

            system_info = get_system_info()
            self._logger.info("=" * 80)
            self._logger.info(f"SESSION START | {START_TIME.strftime(_TIME_FMT)}")
            self._logger.info(f"Application  : {settings.PROJECT_NAME}")
            self._logger.info(f"System Info  : {system_info}")
            self._logger.info("=" * 80)

    @property
    def native(self) -> logging.Logger:
        return self._logger

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, stacklevel=2, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, stacklevel=2, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, stacklevel=2, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, stacklevel=2, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, stacklevel=2, **kwargs)

    def exception(self, msg, *args, **kwargs):
        kwargs.setdefault("exc_info", True)
        self._logger.error(msg, *args, stacklevel=2, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        self._logger.log(level, msg, *args, stacklevel=2, **kwargs)


def exception_logging(exctype, value, tb):
    tb_lines = traceback.format_exception(exctype, value, tb)
    full_traceback = "".join(tb_lines)
    tb_frame = traceback.extract_tb(tb)[-1] if tb else None

    write_val = {
        "exception_type": exctype.__name__,
        "exception_message": str(value),
        "file": tb_frame.filename if tb_frame else "Unknown",
        "line": tb_frame.lineno if tb_frame else "Unknown",
        "function": tb_frame.name if tb_frame else "Unknown",
        "full_traceback": full_traceback,
    }
    logger.error(f"UNHANDLED EXCEPTION: {write_val}")


def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        module_name = func.__module__
        qualified = f"{module_name}.{func_name}"
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)

        start_dt = datetime.now()
        start_ts = time.perf_counter()
        logger.debug(
            f">>> START  {qualified}({signature}) | Start: {start_dt.strftime(_TIME_FMT)}"
        )

        try:
            result = func(*args, **kwargs)
            end_ts = time.perf_counter()
            end_dt = datetime.now()
            duration = end_ts - start_ts
            logger.debug(
                f"<<< END    {qualified} | End: {end_dt.strftime(_TIME_FMT)} | "
                f"Duration: {duration:.4f}s | Result: {repr(result)}"
            )
            return result
        except Exception as e:
            end_ts = time.perf_counter()
            end_dt = datetime.now()
            duration = end_ts - start_ts
            logger.error(
                f"!!! ERROR  {qualified} | End: {end_dt.strftime(_TIME_FMT)} | "
                f"Duration: {duration:.4f}s | Error: {e!r}"
            )
            raise

    return wrapper


def alog_function_call(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        func_name = func.__name__
        module_name = func.__module__
        qualified = f"{module_name}.{func_name}"
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)

        start_dt = datetime.now()
        start_ts = time.perf_counter()
        logger.debug(
            f">>> START  {qualified}({signature}) | Start: {start_dt.strftime(_TIME_FMT)}"
        )

        try:
            result = await func(*args, **kwargs)
            end_ts = time.perf_counter()
            end_dt = datetime.now()
            duration = end_ts - start_ts
            logger.debug(
                f"<<< END    {qualified} | End: {end_dt.strftime(_TIME_FMT)} | "
                f"Duration: {duration:.4f}s | Result: {repr(result)}"
            )
            return result
        except Exception as e:
            end_ts = time.perf_counter()
            end_dt = datetime.now()
            duration = end_ts - start_ts
            logger.error(
                f"!!! ERROR  {qualified} | End: {end_dt.strftime(_TIME_FMT)} | "
                f"Duration: {duration:.4f}s | Error: {e!r}"
            )
            raise

    return wrapper


def log_method_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self_obj = args[0] if args else None
        cls_name = self_obj.__class__.__name__ if self_obj else func.__module__
        qualified = f"{cls_name}.{func.__name__}"

        method_args = args[1:] if args else args
        args_repr = [repr(a) for a in method_args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)

        start_ts = time.perf_counter()
        logger.debug(f">>> START  {qualified}({signature})")

        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_ts
            logger.debug(
                f"<<< END    {qualified} | Duration: {duration:.4f}s | Result: {repr(result)}"
            )
            return result
        except Exception as e:
            duration = time.perf_counter() - start_ts
            logger.error(
                f"!!! ERROR  {qualified} | Duration: {duration:.4f}s | Error: {e!r}"
            )
            raise

    return wrapper


def alog_method_call(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        self_obj = args[0] if args else None
        cls_name = self_obj.__class__.__name__ if self_obj else func.__module__
        qualified = f"{cls_name}.{func.__name__}"

        method_args = args[1:] if args else args
        args_repr = [repr(a) for a in method_args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)

        start_ts = time.perf_counter()
        logger.debug(f">>> START  {qualified}({signature})")

        try:
            result = await func(*args, **kwargs)
            duration = time.perf_counter() - start_ts
            logger.debug(
                f"<<< END    {qualified} | Duration: {duration:.4f}s | Result: {repr(result)}"
            )
            return result
        except Exception as e:
            duration = time.perf_counter() - start_ts
            logger.error(
                f"!!! ERROR  {qualified} | Duration: {duration:.4f}s | Error: {e!r}"
            )
            raise

    return wrapper


class StepTimer:
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.steps: list[tuple[str, float]] = []
        self._pipeline_start = time.perf_counter()

    @contextlib.asynccontextmanager
    async def astep(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - t0
            self.steps.append((name, duration))
            logger.info(f"⏱️  [{self.pipeline_name}] {name}: {duration:.4f}s")

    @contextlib.contextmanager
    def step(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - t0
            self.steps.append((name, duration))
            logger.info(f"⏱️  [{self.pipeline_name}] {name}: {duration:.4f}s")

    def summary(self):
        total = time.perf_counter() - self._pipeline_start
        lines = [f"📊 Pipeline '{self.pipeline_name}' completed in {total:.4f}s"]
        for name, dur in self.steps:
            pct = (dur / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 2)
            lines.append(
                f"   ├─ {name:<35s} {dur:>8.4f}s  ({pct:5.1f}%)  {bar}"
            )
        lines.append(f"   └─ {'TOTAL':<35s} {total:>8.4f}s")
        logger.info("\n".join(lines))


def _on_exit():
    end_time = datetime.now()
    uptime = end_time - START_TIME
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info("=" * 80)
    logger.info(f"SESSION END   | {end_time.strftime(_TIME_FMT)}")
    logger.info(f"SESSION START | {START_TIME.strftime(_TIME_FMT)}")
    logger.info(f"TOTAL UPTIME  | {hours:02d}h {minutes:02d}m {seconds:02d}s")
    logger.info("=" * 80)


logger = CustomLogger(app_name=settings.PROJECT_NAME)
sys.excepthook = exception_logging
atexit.register(_on_exit)

_CONSOLE_FMT = (
    "[%(asctime)s]"
    "[%(levelname)-8s]"
    "[%(location)-50s]"
    " %(message)s"
)


def setup_uvicorn_logging() -> None:
    formatter = ColorFormatter(_CONSOLE_FMT, datefmt=_TIME_FMT)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.addHandler(handler)
        uv_logger.propagate = False


def get_uvicorn_log_config() -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "app": {
                "()": "src.utils.logger_utils.ColorFormatter",
                "fmt": _CONSOLE_FMT,
                "datefmt": _TIME_FMT,
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "app",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }


class RequestTimingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start
            method = scope.get("method", "?")
            path = scope.get("path", "?")
            logger.info(f"🌐 {method} {path} → {status_code} | {duration:.4f}s")
