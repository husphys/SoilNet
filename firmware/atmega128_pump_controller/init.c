//==============================================================
// init.c - minimal ATmega128 initialization
//==============================================================

void init_system(void)
{
    // nRF24L01 software SPI pin directions, preserved from proven project
    DDRA =
        (0<<DDA7) | (0<<DDA6) | (0<<DDA5) |
        (1<<DDA4) | (1<<DDA3) | (1<<DDA2) |
        (1<<DDA1) | (0<<DDA0);

    PORTA =
        (1<<PORTA7) | (1<<PORTA6) | (1<<PORTA5) |
        (0<<PORTA4) | (0<<PORTA3) | (1<<PORTA2) |
        (0<<PORTA1) | (1<<PORTA0);

    PUMP_RELAY_DDR = 1;

#if PUMP_ACTIVE
    PUMP_RELAY = 0;
#else
    PUMP_RELAY = 1;
#endif

    // Timer2 = 1 ms tick using the same values as the proven project.
    // Keep the same CodeVision CPU clock; recalculate if clock changes.
    TCCR2 =
        (0<<WGM20) | (0<<COM21) | (0<<COM20) |
        (0<<WGM21) | (0<<CS22) | (1<<CS21) | (1<<CS20);

    TCNT2 = 0x83;
    OCR2  = 0x00;
    TIMSK = (1<<TOIE2);
}
