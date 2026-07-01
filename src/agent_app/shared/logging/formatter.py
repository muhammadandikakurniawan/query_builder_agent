import json
import logging
import socket
import traceback
from  datetime import datetime, timezone


class JsonFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),

            "host": {
                "hostname": socket.gethostname(),
            },

            "process": {
                "pid": record.process,
                "thread": record.threadName,
            },

            "source": {
                "module": record.module,
                "file": f"{record.pathname}:{record.lineno}",
                "function": record.funcName,
            },
        }

        # custom fields passed via extra={}
        standard = {
            "name", "msg", "args", "levelname",
            "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text",
            "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName",
            "process"
        }

        for key, value in record.__dict__.items():
            if key not in standard:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "stacktrace": "".join(
                    traceback.format_exception(*record.exc_info)
                ),
            }

        return json.dumps(payload, default=str)