//==============================================================
// define.c - SoilNet downstream pump controller
// ATmega128 / CodeVisionAVR
//==============================================================

#define IRQ     PINA.0
#define MOSI    PORTA.1
#define CSN     PORTA.2
#define CE      PORTA.3
#define SCK     PORTA.4
#define MISO    PINA.5

#define NRF_MAX_PAYLOAD      32
// Pump relay output. Change only these if your relay uses another pin.
#define PUMP_RELAY       PORTB.0
#define PUMP_RELAY_DDR   DDRB.0
#define PUMP_ACTIVE      1

#define PUMP_MAX_SECONDS     180U
