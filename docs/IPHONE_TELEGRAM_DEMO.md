# SoilNet iPhone → Telegram → Raspberry Pi demo

## What BotFather is

The text “BotFather is the one bot to rule them all” is Telegram's description
of `@BotFather`. BotFather manages bot accounts; it is not the SoilNet bot.

Create a new SoilNet bot in Telegram:

1. Open the verified `@BotFather`.
2. Send `/newbot`.
3. Choose a display name, for example `SoilNet Irrigation`.
4. Choose a unique username ending in `bot`.
5. BotFather returns an API token.
6. Put the token only in the Pi environment file. Do not store it in source code.

If an old bot still exists under `/mybots`, you may reuse it and rotate its token.

## Intended experiment

iPhone RGB image
→ Telegram bot
→ Raspberry Pi
→ frozen `P1_SOILNET_VICREG_MU27_NO_LI_BESTREG`
→ SM-0 and SM-20 predictions
→ zero-order Sugeno fuzzy controller
→ nRF24L01
→ downstream controller
→ locally timed pump actuation.

No light intensity input is used in this deployment branch.

This is a prospective deployment demonstration. It does not retrain SoilNet,
select a new checkpoint, access the locked final test set, or alter final paper
results.

## Frozen model contract

Expected checkpoint:

`P1_SOILNET_VICREG_MU27_NO_LI_BESTREG`

Expected SHA256:

`a9d8de995b0673e9ec39bfc4afac00d6a2a773ad9ffbea0027ff2d0c4fab820c`

The supplied loader:

- verifies SHA256;
- verifies the experiment ID;
- verifies `li_consumed_by_model=False`;
- loads the state dict with `strict=True`;
- converts images to RGB;
- resizes directly to `224×224`;
- normalizes using ImageNet mean/std;
- calls `model(image, None)`;
- multiplies regression outputs by 100 exactly once.

## Install on Raspberry Pi

Assuming the repo is `/home/pi/SoilNet`:

```bash
cd /home/pi/SoilNet
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements-rpi-demo.txt
```

Enable SPI in `raspi-config`.

If needed:

```bash
sudo usermod -aG spi,gpio pi
```

Reboot after group changes.

## Provision checkpoint

Example:

```bash
sudo mkdir -p /opt/soilnet/models
sudo cp validation_best_regression.pth \
  /opt/soilnet/models/P1_noLI_validation_best_regression.pth

sha256sum /opt/soilnet/models/P1_noLI_validation_best_regression.pth
```

Stop if the SHA does not match the expected frozen checkpoint.

## Configure Telegram

Create:

```bash
sudo mkdir -p /etc/soilnet
sudo nano /etc/soilnet/iphone-demo.env
```

Use `.env.example` as the template.

For initial setup, start the bot, send `/whoami`, note the returned chat ID, stop
the service, then place that ID in `TELEGRAM_ALLOWED_CHAT_IDS`.

Only allowlisted chats can submit images for inference.

## First test: no radio, no pump

Keep:

```yaml
radio:
  enabled: false
```

Do not set `SOILNET_ENABLE_PUMP`.

Run:

```bash
cd /home/pi/SoilNet
source .venv/bin/activate
set -a
source /etc/soilnet/iphone-demo.env
set +a
python scripts/check_iphone_demo.py
python scripts/run_iphone_telegram_demo.py
```

Send a soil photo from the iPhone to the bot.

Expected reply:

- SM-0 prediction
- SM-20 prediction
- fuzzy irrigation duration
- inference latency
- end-to-end latency
- `Mode: DRY RUN`

## Fuzzy controller

Inputs:

- predicted SM-0
- predicted SM-20

No LI is used.

Current provisional membership functions:

- Dry = trapezoid `[0, 0, 30, 45]`
- Moderate = triangle `[30, 50, 70]`
- Wet = trapezoid `[55, 70, 100, 100]`

Current singleton rule matrix in minutes:

| SM-0 \ SM-20 | Dry | Moderate | Wet |
|---|---:|---:|---:|
| Dry | 3 | 2 | 1 |
| Moderate | 2 | 1 | 0 |
| Wet | 1 | 0 | 0 |

Product firing strength:

`w_ij = mu_i(SM0) * mu_j(SM20)`

Zero-order Sugeno output:

`T = sum(w_ij * t_ij) / sum(w_ij)`

The result is capped at 180 seconds.

These values are provisional engineering choices. Freeze the policy before the
prospective demonstration and do not tune it from the new iPhone images.

## nRF24L01 command strategy

The Pi transmits an irrigation duration, not a persistent ON command.

Example:

`RUN 84 s`

The downstream controller turns the pump on and maintains its own local timer.
It turns the pump off when the duration expires even if the Pi or radio link is
lost.

Safety behavior in the supplied firmware:

- default relay state OFF;
- maximum run duration 180 s;
- CRC16 validation;
- session ID and message ID;
- duplicate commands do not restart or extend irrigation;
- application response packet is sent back to Pi;
- malformed or over-limit commands are rejected.

## Bring-up order

Do not connect the water pump first.

1. Telegram + model only.
2. Inspect fuzzy outputs in dry-run mode.
3. Test nRF24 sender and receiver with no relay.
4. Test relay with an LED or low-voltage dummy load.
5. Verify relay polarity.
6. Verify local automatic OFF at the downstream controller.
7. Verify receiver reboot always starts OFF.
8. Connect the actual pump only after these tests pass.
9. Change `radio.enabled` to `true`.
10. Set:

```bash
SOILNET_ENABLE_PUMP=YES_I_HAVE_VERIFIED_HARDWARE
```

Both the YAML radio flag and this exact environment phrase are required before
the Pi sends pump commands.

## Default downstream pins

Supplied Arduino-style firmware defaults:

- nRF24 CE = D9
- nRF24 CSN = D10
- relay signal = D7
- SPI pins = board hardware SPI pins

nRF24L01 uses 3.3 V. Do not apply 5 V to the radio.

The pump must not be driven directly from a microcontroller pin. Use a suitable
relay or driver and an external pump power supply.

## Logs for the paper

The JSONL log records:

- Telegram reception
- image SHA256 and size
- image dimensions
- frozen checkpoint ID/SHA
- raw SM-0 and SM-20
- controller-clamped values
- inference latency
- fuzzy memberships and firing strengths
- fuzzy irrigation duration
- radio attempts and application response
- downstream reported pump state
- end-to-end latency
- failures

Downloaded Telegram images are deleted after processing by default.

A prospective deployment experiment can summarize:

- number of iPhone images sent
- image reception success rate
- inference success rate
- end-to-end latency
- fuzzy duration distribution
- radio delivery success
- pump actuation success

Do not call this an external accuracy validation unless independent SM-0/SM-20
ground truth is measured under a separate pre-specified protocol.

## Run software tests

```bash
pytest -q tests/test_deployment_fuzzy_protocol.py
```

## Run as a service

Edit paths/user in `deployment/systemd/soilnet-iphone-demo.service` if needed.

Then:

```bash
sudo cp deployment/systemd/soilnet-iphone-demo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now soilnet-iphone-demo
sudo systemctl status soilnet-iphone-demo
journalctl -u soilnet-iphone-demo -f
```
