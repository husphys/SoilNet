from __future__ import annotations

import struct
from dataclasses import dataclass

PACKET_SIZE = 32
VERSION = 1
CMD_MAGIC = b"SN"
RSP_MAGIC = b"SR"
CMD_OFF = 0
CMD_RUN = 1
RSP_ACCEPTED = 1
RSP_DUPLICATE = 2
RSP_REJECTED = 3


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def build_command(session_id: int, message_id: int, duration_seconds: int) -> bytes:
    duration_seconds = min(65535, max(0, int(duration_seconds)))
    command = CMD_RUN if duration_seconds > 0 else CMD_OFF
    header = struct.pack(
        "<2sBBIIH",
        CMD_MAGIC, VERSION, command,
        session_id & 0xFFFFFFFF,
        message_id & 0xFFFFFFFF,
        duration_seconds,
    )
    packet = header + struct.pack("<H", crc16_ccitt(header))
    return packet.ljust(PACKET_SIZE, b"\x00")


@dataclass(frozen=True)
class Response:
    status: int
    session_id: int
    message_id: int
    accepted_duration_seconds: int
    pump_state: int


def parse_response(packet: bytes) -> Response:
    if len(packet) < 18:
        raise ValueError("Response packet too short.")
    payload = packet[:16]
    received_crc = struct.unpack("<H", packet[16:18])[0]
    if crc16_ccitt(payload) != received_crc:
        raise ValueError("Response CRC mismatch.")
    magic, version, status, session_id, message_id, duration, pump_state, _reserved = struct.unpack(
        "<2sBBIIHBB", payload
    )
    if magic != RSP_MAGIC or version != VERSION:
        raise ValueError("Response magic/version mismatch.")
    return Response(status, session_id, message_id, duration, pump_state)
