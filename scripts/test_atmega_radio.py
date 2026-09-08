#!/usr/bin/env python3
"""
Direct Raspberry Pi <-> ATmega128 nRF24L01 test for SoilNet.

Expected ATmega firmware:
  permanent PRX + ACK payload
  address A1 A1 A1 A1 A1
  channel 2
  1 Mbps
  CRC16
  dynamic payload + ACK payload

Default Raspberry Pi wiring:
  CE  = GPIO22
  CSN = SPI0 CE0 (/dev/spidev0.0)

Usage:
  python test_atmega_radio.py ping
  python test_atmega_radio.py status
  python test_atmega_radio.py run 5
  python test_atmega_radio.py off
"""

import argparse
import random
import time

from pyrf24 import RF24, RF24_DRIVER, RF24_PA_LOW

ADDRESS = b"\xA1" * 5
CHANNEL = 2
CSN_PIN = 0  # /dev/spidev0.0

if RF24_DRIVER == "MRAA":
    CE_PIN = 15  # maps to GPIO22
elif RF24_DRIVER == "wiringPi":
    CE_PIN = 3   # maps to GPIO22
else:
    CE_PIN = 22


def read_ack_payload(radio):
    if not radio.available():
        return None

    width = radio.get_dynamic_payload_size()
    if width < 1 or width > 32:
        return None

    data = bytes(radio.read(width))
    try:
        return data.decode("ascii", errors="replace")
    except Exception:
        return repr(data)


def send_packet(radio, payload: bytes):
    ok = bool(radio.write(payload))
    ack = read_ack_payload(radio)
    return ok, ack


def command_then_poll(radio, command: str):
    # The AVR prepares the command response AFTER the hardware ACK has already
    # been transmitted. Therefore the response is retrieved by a second packet.
    ok1, ack1 = send_packet(radio, command.encode("ascii"))
    print(f"TX   : {command}")
    print(f"LINK : {'ACK' if ok1 else 'NO ACK'}")
    print(f"ACK1 : {ack1 if ack1 is not None else '<empty>'}")

    if not ok1:
        return False

    time.sleep(0.08)

    ok2, ack2 = send_packet(radio, b"POLL")
    print("TX   : POLL")
    print(f"LINK : {'ACK' if ok2 else 'NO ACK'}")
    print(f"ACK2 : {ack2 if ack2 is not None else '<empty>'}")

    return ok2


def make_radio():
    radio = RF24(CE_PIN, CSN_PIN)

    if not radio.begin():
        raise RuntimeError(
            "nRF24L01 on Raspberry Pi is not responding. "
            "Check SPI, 3.3 V power, CE/CSN/SCK/MOSI/MISO wiring."
        )

    # Match the ATmega128 receiver.
    radio.set_pa_level(RF24_PA_LOW)
    radio.channel = CHANNEL
    radio.set_retries(5, 15)
    radio.dynamic_payloads = True
    radio.ack_payloads = True

    # Pi is permanent PTX. This also configures TX pipe 0.
    radio.stop_listening(ADDRESS)

    print("Pi nRF24 initialized")
    print(f"driver={RF24_DRIVER}, CE={CE_PIN}, CSN={CSN_PIN}, channel={CHANNEL}")
    print("address=A1 A1 A1 A1 A1")
    return radio


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping")
    sub.add_parser("status")
    sub.add_parser("off")

    run_p = sub.add_parser("run")
    run_p.add_argument("seconds", type=int)

    args = parser.parse_args()
    radio = make_radio()

    # Short session/message IDs fit comfortably in the <=32-byte AVR payload.
    session = random.randint(1, 60000)
    message = 1

    if args.cmd == "ping":
        command_then_poll(radio, "PING")
        return

    if args.cmd == "status":
        command_then_poll(radio, "STATUS")
        return

    if args.cmd == "run":
        seconds = max(1, min(180, args.seconds))
        command_then_poll(radio, f"RUN,{session},{message},{seconds}")
        return

    if args.cmd == "off":
        command_then_poll(radio, f"OFF,{session},{message}")
        return


if __name__ == "__main__":
    main()
