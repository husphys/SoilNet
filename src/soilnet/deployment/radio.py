from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from pyrf24 import RF24, RF24_DRIVER, RF24_PA_LOW


ADDRESS = b"\xA1" * 5


@dataclass(frozen=True)
class RadioResult:
    success: bool
    attempts: int
    status: int | None
    accepted_duration_seconds: int | None
    pump_state: int | None
    detail: str


class NRF24IrrigationLink:
    """
    SoilNet Raspberry Pi -> ATmega128 nRF24L01 link.

    Matches the already hardware-verified ATmega128 firmware:
      Pi        = permanent PTX
      ATmega128 = permanent PRX + ACK payload
      address   = A1 A1 A1 A1 A1
      channel   = 2
      data rate = 1 Mbps
      dynamic payload + ACK payload

    Application protocol:
      RUN,<session>,<message_id>,<seconds>
      OFF,<session>,<message_id>
      POLL

    The adapter accepts either:
      1) the application response arriving in ACK1 immediately, or
      2) a queued response retrieved by a subsequent POLL.
    This makes it robust to the ACK-payload behavior observed on the real setup.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.session_id = random.SystemRandom().randrange(1, 60001)
        self.message_id = 0
        self.radio: RF24 | None = None

        if self.enabled:
            self._start()

    def _start(self) -> None:
        ce_pin = int(self.config.get("ce_pin", 22))
        csn_pin = int(self.config.get("csn_pin", 0))

        # Match the working standalone test script across pyrf24 backends.
        if RF24_DRIVER == "MRAA" and ce_pin == 22:
            ce_pin = 15
        elif RF24_DRIVER == "wiringPi" and ce_pin == 22:
            ce_pin = 3

        radio = RF24(ce_pin, csn_pin)

        if not radio.begin():
            raise RuntimeError(
                "nRF24L01 initialization failed on Raspberry Pi. "
                "Check SPI and CE/CSN wiring."
            )

        radio.set_pa_level(RF24_PA_LOW)
        radio.channel = int(self.config.get("channel", 2))
        radio.set_retries(5, 15)
        radio.dynamic_payloads = True
        radio.ack_payloads = True

        # Permanent PTX. This configures the writing pipe and ACK pipe 0.
        radio.stop_listening(ADDRESS)

        self.radio = radio

    def _next_message_id(self) -> int:
        self.message_id += 1
        if self.message_id > 60000:
            self.message_id = 1
        return self.message_id

    @staticmethod
    def _decode_payload(payload: bytes | bytearray | list[int] | None) -> str | None:
        if payload is None:
            return None
        data = bytes(payload)
        if not data:
            return None
        return data.decode("ascii", errors="replace").strip("\x00\r\n ")

    def _read_ack_payload(self) -> str | None:
        assert self.radio is not None

        if not self.radio.available():
            return None

        width = int(self.radio.get_dynamic_payload_size())
        if width < 1 or width > 32:
            return None

        return self._decode_payload(self.radio.read(width))

    def _write(self, text: str) -> tuple[bool, str | None]:
        assert self.radio is not None
        ok = bool(self.radio.write(text.encode("ascii")))
        ack = self._read_ack_payload()
        return ok, ack

    @staticmethod
    def _response_matches(
        response: str | None,
        session_id: int,
        message_id: int,
        duration_seconds: int,
    ) -> tuple[bool, str]:
        if not response:
            return False, "EMPTY_ACK"

        if duration_seconds > 0:
            expected = f"ACK,RUN,{session_id},{message_id},{duration_seconds}"
        else:
            expected = f"ACK,OFF,{session_id},{message_id}"

        duplicate = f"DUP,{session_id},{message_id}"

        if response == expected:
            return True, "ACCEPTED"

        if response == duplicate:
            return True, "DUPLICATE_ACCEPTED"

        if response == "ID_CONFLICT":
            return False, "ID_CONFLICT"

        if response in ("BAD_CMD", "BAD_TIME", "UNKNOWN"):
            return False, response

        return False, f"UNRELATED_ACK:{response}"

    def send_duration(self, duration_seconds: int) -> RadioResult:
        duration_seconds = max(0, min(180, int(duration_seconds)))
        message_id = self._next_message_id()

        if not self.enabled:
            return RadioResult(
                True,
                0,
                None,
                duration_seconds,
                None,
                "RADIO_DISABLED_DRY_RUN",
            )

        assert self.radio is not None

        if duration_seconds > 0:
            command = (
                f"RUN,{self.session_id},{message_id},{duration_seconds}"
            )
        else:
            command = f"OFF,{self.session_id},{message_id}"

        retries = max(1, int(self.config.get("application_retries", 3)))
        poll_delay_s = max(
            0.02,
            float(self.config.get("poll_delay_ms", 80)) / 1000.0,
        )

        last_detail = "NO_ACK"

        for attempt in range(1, retries + 1):
            link_ok, ack1 = self._write(command)

            if not link_ok:
                last_detail = "NO_RF_ACK"
                continue

            matched, detail = self._response_matches(
                ack1,
                self.session_id,
                message_id,
                duration_seconds,
            )

            if matched:
                return RadioResult(
                    True,
                    attempt,
                    1,
                    duration_seconds,
                    1 if duration_seconds > 0 else 0,
                    f"ACK1_{detail}",
                )

            # If ACK1 was READY, empty, or a previous response, retrieve the
            # response queued by the command using POLL.
            time.sleep(poll_delay_s)

            poll_ok, ack2 = self._write("POLL")

            if not poll_ok:
                last_detail = f"{detail};POLL_NO_RF_ACK"
                continue

            matched, detail2 = self._response_matches(
                ack2,
                self.session_id,
                message_id,
                duration_seconds,
            )

            if matched:
                return RadioResult(
                    True,
                    attempt,
                    1,
                    duration_seconds,
                    1 if duration_seconds > 0 else 0,
                    f"ACK2_{detail2}",
                )

            last_detail = f"ACK1={ack1!r};ACK2={ack2!r};{detail2}"

        return RadioResult(
            False,
            retries,
            None,
            None,
            None,
            f"DELIVERY_UNKNOWN:{last_detail}",
        )
