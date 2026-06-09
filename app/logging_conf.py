from pathlib import Path
import logging
import os
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk.resources import Resource
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("HYPERDX_API_KEY")

def get_logger(name = "churn-api") -> logging.Logger :
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate= False 

    handler= logging.StreamHandler()
    formatter= logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    log_path= Path(__file__).parent / "app.log"
    file_handler= logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # --- HyperDX (OpenTelemetry) handler ---
    if api_key:
        provider = LoggerProvider(resource=Resource.create({"service.name": name}))
        exporter = OTLPLogExporter(
            endpoint="https://in-otel.hyperdx.io/v1/logs",
            headers={"authorization": api_key},
        )
        provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        otel_handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
        logger.addHandler(otel_handler)
        logger.info("HyperDX logging enabled")

    return logger