from soilnet.deployment.fuzzy import SugenoIrrigationController
from soilnet.deployment.protocol import build_command, crc16_ccitt

FUZZY = {
    "dry": {"type": "trapezoid", "points": [0, 0, 30, 45]},
    "moderate": {"type": "triangle", "points": [30, 50, 70]},
    "wet": {"type": "trapezoid", "points": [55, 70, 100, 100]},
    "rules_minutes": {
        "dry": {"dry": 3.0, "moderate": 2.0, "wet": 1.0},
        "moderate": {"dry": 2.0, "moderate": 1.0, "wet": 0.0},
        "wet": {"dry": 1.0, "moderate": 0.0, "wet": 0.0},
    },
    "maximum_duration_seconds": 180,
}


def test_dry_dry_is_maximum():
    assert SugenoIrrigationController(FUZZY).evaluate(10, 10).duration_seconds == 180


def test_wet_wet_is_off():
    assert SugenoIrrigationController(FUZZY).evaluate(90, 90).duration_seconds == 0


def test_command_packet_crc():
    packet = build_command(0x12345678, 9, 120)
    assert len(packet) == 32
    assert crc16_ccitt(packet[:14]) == int.from_bytes(packet[14:16], "little")
