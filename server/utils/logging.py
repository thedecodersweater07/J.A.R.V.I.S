import logging

def configure_logging(settings) -> logging.Logger:
    logging.basicConfig(
        level=settings.log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("jarvis.server")
