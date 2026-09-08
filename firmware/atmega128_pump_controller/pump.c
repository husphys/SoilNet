//==============================================================
// pump.c - non-blocking local pump timer
//==============================================================

volatile unsigned long system_ms = 0;

unsigned char pump_running = 0;
unsigned long pump_start_ms = 0;
unsigned long pump_duration_ms = 0;

interrupt [TIM2_OVF] void timer2_ovf_isr(void)
{
    TCNT2 = 0x83;
    system_ms++;
}

unsigned long get_millis(void)
{
    unsigned long value;

    #asm("cli")
    value = system_ms;
    #asm("sei")

    return value;
}

void pump_hw_on(void)
{
#if PUMP_ACTIVE
    PUMP_RELAY = 1;
#else
    PUMP_RELAY = 0;
#endif
}

void pump_hw_off(void)
{
#if PUMP_ACTIVE
    PUMP_RELAY = 0;
#else
    PUMP_RELAY = 1;
#endif
}

void pump_stop(void)
{
    pump_hw_off();
    pump_running = 0;
    pump_start_ms = 0;
    pump_duration_ms = 0;
}

void pump_start_seconds(unsigned int seconds)
{
    if(seconds == 0)
    {
        pump_stop();
        return;
    }

    if(seconds > PUMP_MAX_SECONDS)
    {
        seconds = PUMP_MAX_SECONDS;
    }

    pump_start_ms = get_millis();
    pump_duration_ms = (unsigned long)seconds * 1000UL;

    pump_hw_on();
    pump_running = 1;
}

void pump_task(void)
{
    unsigned long now;

    if(!pump_running)
    {
        return;
    }

    now = get_millis();

    if(
        (unsigned long)(now - pump_start_ms)
        >=
        pump_duration_ms
    )
    {
        pump_stop();
    }
}

unsigned int pump_remaining_seconds(void)
{
    unsigned long now;
    unsigned long elapsed;
    unsigned long remaining;

    if(!pump_running)
    {
        return 0;
    }

    now = get_millis();
    elapsed = (unsigned long)(now - pump_start_ms);

    if(elapsed >= pump_duration_ms)
    {
        return 0;
    }

    remaining = pump_duration_ms - elapsed;

    return (unsigned int)((remaining + 999UL) / 1000UL);
}
