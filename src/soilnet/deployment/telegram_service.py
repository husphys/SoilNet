from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .event_log import EventLogger
from .fuzzy import SugenoIrrigationController
from .model import FrozenNoLISoilNet
from .radio import NRF24IrrigationLink


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _allowed_chat_ids() -> set[int]:
    raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").strip()
    if not raw:
        return set()
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


class SoilNetTelegramService:
    def __init__(self, config_path: str | Path) -> None:
        config_path = Path(config_path).expanduser().resolve()
        self.config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        checkpoint = os.environ.get("SOILNET_P1_CHECKPOINT", "").strip()
        log_dir = os.environ.get("SOILNET_DEMO_LOG_DIR", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required.")
        if not checkpoint:
            raise RuntimeError("SOILNET_P1_CHECKPOINT is required.")
        if not log_dir:
            raise RuntimeError("SOILNET_DEMO_LOG_DIR is required.")

        self.token = token
        self.allowed = _allowed_chat_ids()
        self.processing_lock = asyncio.Lock()

        m = self.config["model"]
        self.model = FrozenNoLISoilNet(
            checkpoint,
            expected_sha256=m["expected_checkpoint_sha256"],
            expected_experiment_id=m["expected_experiment_id"],
            num_classes=int(m["num_classes"]),
            image_size=tuple(m["image_size"]),
            normalization_mean=tuple(m["normalization_mean"]),
            normalization_std=tuple(m["normalization_std"]),
            torch_threads=int(m.get("torch_threads", 4)),
            device="cpu",
        )
        self.fuzzy = SugenoIrrigationController(self.config["fuzzy"])
        self.radio = NRF24IrrigationLink(self.config["radio"])
        self.logger = EventLogger(Path(log_dir) / self.config["logging"]["filename"])

        self.pump_armed = (
            bool(self.config["radio"].get("enabled", False))
            and os.environ.get("SOILNET_ENABLE_PUMP", "") == "YES_I_HAVE_VERIFIED_HARDWARE"
        )

    def _is_allowed(self, update: Update) -> bool:
        chat = update.effective_chat
        return bool(chat and chat.id in self.allowed)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            "SoilNet smartphone demo is online.\n"
            "Send one soil photo. The deployment uses the frozen no-LI model.\n"
            "Use /whoami to display this chat ID."
        )

    async def whoami(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat:
            await update.effective_message.reply_text(f"Chat ID: {update.effective_chat.id}")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            await update.effective_message.reply_text("This chat is not authorized.")
            return
        await update.effective_message.reply_text(
            f"Model: {self.model.experiment_id}\n"
            f"Epoch: {self.model.checkpoint_epoch}\n"
            f"Checkpoint: {self.model.checkpoint_sha256[:12]}...\n"
            f"Radio configured: {self.radio.enabled}\n"
            f"Pump armed: {self.pump_armed}"
        )

    async def _download_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE, target: Path) -> None:
        message = update.effective_message
        if message.photo:
            tg_file = await context.bot.get_file(message.photo[-1].file_id)
        elif message.document and (message.document.mime_type or "").startswith("image/"):
            tg_file = await context.bot.get_file(message.document.file_id)
        else:
            raise ValueError("No supported image found.")
        await tg_file.download_to_drive(custom_path=str(target))

    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not self._is_allowed(update):
            await message.reply_text(
                "This chat is not authorized. Use /whoami and add the ID to TELEGRAM_ALLOWED_CHAT_IDS."
            )
            return

        async with self.processing_lock:
            correlation_id = uuid.uuid4().hex
            t0 = time.perf_counter_ns()
            self.logger.write("telegram_received", correlation_id)

            suffix = ".jpg"
            if message.document and message.document.file_name:
                suffix = Path(message.document.file_name).suffix or ".img"

            fd, tmp_name = tempfile.mkstemp(prefix="soilnet_", suffix=suffix)
            os.close(fd)
            tmp_path = Path(tmp_name)

            try:
                await self._download_image(update, context, tmp_path)
                raw = tmp_path.read_bytes()
                self.logger.write(
                    "image_downloaded",
                    correlation_id,
                    image_sha256=_sha256_bytes(raw),
                    image_bytes=len(raw),
                )

                decode_start = time.perf_counter_ns()
                try:
                    with Image.open(tmp_path) as image:
                        width, height = image.size
                        image.load()
                except UnidentifiedImageError:
                    self.logger.write("image_decode_failed", correlation_id)
                    await message.reply_text(
                        "Unsupported image. Send the iPhone image as a Telegram Photo/JPEG."
                    )
                    return

                self.logger.write(
                    "image_decoded",
                    correlation_id,
                    width=width,
                    height=height,
                    decode_latency_ms=(time.perf_counter_ns() - decode_start) / 1e6,
                )

                infer_start = time.perf_counter_ns()
                prediction = self.model.predict_path(tmp_path)
                infer_ms = (time.perf_counter_ns() - infer_start) / 1e6
                self.logger.write(
                    "inference_completed",
                    correlation_id,
                    inference_latency_ms=infer_ms,
                    experiment_id=self.model.experiment_id,
                    checkpoint_sha256=self.model.checkpoint_sha256,
                    **prediction,
                )

                fuzzy_start = time.perf_counter_ns()
                decision = self.fuzzy.evaluate(
                    prediction["sm0_controller"],
                    prediction["sm20_controller"],
                )
                fuzzy_ms = (time.perf_counter_ns() - fuzzy_start) / 1e6
                self.logger.write(
                    "fuzzy_evaluated",
                    correlation_id,
                    fuzzy_latency_ms=fuzzy_ms,
                    duration_seconds=decision.duration_seconds,
                    duration_minutes_raw=decision.duration_minutes_raw,
                    memberships_sm0=decision.memberships_sm0,
                    memberships_sm20=decision.memberships_sm20,
                    firing_strengths=decision.firing_strengths,
                )

                radio_result = None
                radio_ms = 0.0
                if self.pump_armed:
                    radio_start = time.perf_counter_ns()
                    radio_result = self.radio.send_duration(decision.duration_seconds)
                    radio_ms = (time.perf_counter_ns() - radio_start) / 1e6
                    self.logger.write(
                        "radio_completed",
                        correlation_id,
                        radio_latency_ms=radio_ms,
                        success=radio_result.success,
                        attempts=radio_result.attempts,
                        response_status=radio_result.status,
                        accepted_duration_seconds=radio_result.accepted_duration_seconds,
                        pump_state=radio_result.pump_state,
                        detail=radio_result.detail,
                    )

                end_to_end_ms = (time.perf_counter_ns() - t0) / 1e6
                self.logger.write(
                    "pipeline_completed",
                    correlation_id,
                    end_to_end_latency_ms=end_to_end_ms,
                    pump_armed=self.pump_armed,
                )

                mode = "PUMP ARMED" if self.pump_armed else "DRY RUN"
                radio_text = ""
                if radio_result is not None:
                    radio_text = (
                        f"\nRadio delivery: {'OK' if radio_result.success else 'FAILED/UNKNOWN'}"
                        f"\nAttempts: {radio_result.attempts}"
                    )

                await message.reply_text(
                    f"SoilNet no-LI result\n"
                    f"SM-0: {prediction['sm0_raw']:.2f}%\n"
                    f"SM-20: {prediction['sm20_raw']:.2f}%\n"
                    f"Fuzzy irrigation duration: {decision.duration_seconds} s\n"
                    f"Inference: {infer_ms:.1f} ms\n"
                    f"End to end: {end_to_end_ms:.1f} ms\n"
                    f"Mode: {mode}"
                    f"{radio_text}"
                )

            except Exception as exc:
                self.logger.write(
                    "pipeline_failed",
                    correlation_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                await message.reply_text(
                    f"Processing failed: {type(exc).__name__}. Check the Pi service log."
                )
            finally:
                if self.config["telegram"].get("delete_downloaded_images", True):
                    tmp_path.unlink(missing_ok=True)

    def run(self) -> None:
        app = Application.builder().token(self.token).build()
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("whoami", self.whoami))
        app.add_handler(CommandHandler("status", self.status))
        app.add_handler(MessageHandler(filters.PHOTO, self.handle_image))
        app.add_handler(MessageHandler(filters.Document.IMAGE, self.handle_image))
        app.run_polling(allowed_updates=Update.ALL_TYPES)


def main() -> None:
    config_path = os.environ.get("SOILNET_DEMO_CONFIG", "config/deployment/iphone_demo.yaml")
    SoilNetTelegramService(config_path).run()


if __name__ == "__main__":
    main()
