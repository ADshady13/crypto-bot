"""
Notification — Telegram alerts (optional stub).

Sends trade alerts to a Telegram chat. If credentials are not configured,
all calls silently no-op so the bot runs without Telegram.
"""

import logging
import requests
from core.config import Config

logger = logging.getLogger("crypto_bot")


class Notifier:
    """Sends Telegram messages. Silently no-ops if not configured."""

    def __init__(self):
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        if self.enabled:
            logger.info("Telegram notifications: ENABLED")
        else:
            logger.info("Telegram notifications: DISABLED (no token/chat_id)")

    def send(self, message: str):
        """Send a message. No-ops if not configured."""
        if not self.enabled:
            return

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"Telegram API error: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")

    def trade_alert(self, direction: str, pair: str, price: float, reason: str):
        """Send a formatted trade alert."""
        emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "⚪"
        msg = (
            f"{emoji} *{direction}* {pair}\n"
            f"Price: `${price:,.2f}`\n"
            f"Reason: {reason}\n"
            f"Mode: {Config.TRADE_MODE}"
        )
        self.send(msg)
        logger.info(f"Alert sent: {direction} {pair} @ {price}")

    def model_alert(self, message: str):
        """Send a model retraining alert."""
        self.send(f"🤖 *Model Update*\n{message}")
