import logging

logging.basicConfig(
    level=logging.INFO,  # Ensure this is INFO or lower
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("chat-assistant")
logger.setLevel(logging.INFO)
