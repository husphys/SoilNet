//==============================================================
// main.c - SoilNet downstream irrigation pump controller
// ATmega128 / CodeVisionAVR
//
// Radio profile preserved from the previously working project:
//   Pi       = permanent PTX
//   ATmega128= permanent PRX + ACK Payload
//   Address  = A1 A1 A1 A1 A1
//   Channel  = 2
//   1 Mbps
//   CRC16
//   Dynamic Payload ON
//   ACK Payload ON
//
// Application commands:
//   PING
//   STATUS
//   RUN,<session>,<id>,<seconds>
//   OFF,<session>,<id>
//   POLL
//
// ACK payload is pipelined:
//   Pi sends command -> receives previous ACK payload
//   Pi sends POLL    -> receives response queued by command
//
// Safety:
//   - pump OFF at boot
//   - max run time = 180 s
//   - local ATmega timer turns pump OFF even if Pi/radio is lost
//   - duplicate session/id does NOT restart the timer
//==============================================================

#include <mega128.h>
#include <delay.h>
#include <stdio.h>
#include <stdlib.h>

#include "define.c"
#include "init.c"
#include "pump.c"
#include "nrf_24l01.c"


#define APP_CMD_NONE     0
#define APP_CMD_RUN      1
#define APP_CMD_OFF      2


unsigned char have_last_command = 0;
unsigned int last_session_id = 0;
unsigned int last_message_id = 0;
unsigned char last_command_type = APP_CMD_NONE;
unsigned int last_duration_seconds = 0;


//==============================================================
// RAM STRING HELPERS
//==============================================================

unsigned char text_equal(char *a, char *b)
{
    unsigned char i = 0;

    while((a[i] != 0) && (b[i] != 0))
    {
        if(a[i] != b[i])
        {
            return 0;
        }

        i++;
    }

    if((a[i] == 0) && (b[i] == 0))
    {
        return 1;
    }

    return 0;
}


unsigned char starts_with(char *text, char *prefix)
{
    unsigned char i = 0;

    while(prefix[i] != 0)
    {
        if(text[i] != prefix[i])
        {
            return 0;
        }

        i++;
    }

    return 1;
}


unsigned int parse_uint(char *text, unsigned char *index)
{
    unsigned long value = 0;

    while(
        (text[*index] >= '0') &&
        (text[*index] <= '9')
    )
    {
        value =
            value * 10UL +
            (unsigned int)(text[*index] - '0');

        (*index)++;

        if(value > 65535UL)
        {
            value = 65535UL;
        }
    }

    return (unsigned int)value;
}


unsigned char command_equal(char *cmd, unsigned char command_id)
{
    char pattern[12];

    switch(command_id)
    {
        case 1:
            sprintf(pattern, "PING");
            break;

        case 2:
            sprintf(pattern, "STATUS");
            break;

        case 3:
            sprintf(pattern, "POLL");
            break;

        default:
            sprintf(pattern, "");
            break;
    }

    return text_equal(cmd, pattern);
}


unsigned char command_starts_run(char *cmd)
{
    char pattern[8];

    sprintf(pattern, "RUN,");

    return starts_with(cmd, pattern);
}


unsigned char command_starts_off(char *cmd)
{
    char pattern[8];

    sprintf(pattern, "OFF,");

    return starts_with(cmd, pattern);
}


//==============================================================
// DUPLICATE / CONFLICT CHECK
//==============================================================

unsigned char is_same_message(
    unsigned int session_id,
    unsigned int message_id
)
{
    if(
        have_last_command &&
        (session_id == last_session_id) &&
        (message_id == last_message_id)
    )
    {
        return 1;
    }

    return 0;
}


void remember_command(
    unsigned int session_id,
    unsigned int message_id,
    unsigned char command_type,
    unsigned int duration_seconds
)
{
    have_last_command = 1;

    last_session_id = session_id;
    last_message_id = message_id;
    last_command_type = command_type;
    last_duration_seconds = duration_seconds;
}


//==============================================================
// STATUS RESPONSE
//==============================================================

void prepare_status_response(void)
{
    char buff[NRF_MAX_PAYLOAD + 1];
    unsigned int remaining;

    if(pump_running)
    {
        remaining = pump_remaining_seconds();

        sprintf(
            buff,
            "S,ON,%u",
            remaining
        );
    }
    else
    {
        sprintf(
            buff,
            "S,OFF,0"
        );
    }

    NRF_Set_ACK_Text(buff);
}


//==============================================================
// RUN COMMAND
//
// RUN,<session>,<id>,<seconds>
// Example:
//   RUN,4711,12,180
//==============================================================

void handle_run_command(char *cmd)
{
    char buff[NRF_MAX_PAYLOAD + 1];

    unsigned char index = 4;

    unsigned int session_id;
    unsigned int message_id;
    unsigned int seconds;

    session_id = parse_uint(cmd, &index);

    if(cmd[index] != ',')
    {
        sprintf(buff, "BAD_CMD");
        NRF_Set_ACK_Text(buff);
        return;
    }

    index++;

    message_id = parse_uint(cmd, &index);

    if(cmd[index] != ',')
    {
        sprintf(buff, "BAD_CMD");
        NRF_Set_ACK_Text(buff);
        return;
    }

    index++;

    seconds = parse_uint(cmd, &index);

    if(cmd[index] != 0)
    {
        sprintf(buff, "BAD_CMD");
        NRF_Set_ACK_Text(buff);
        return;
    }

    if(
        (seconds == 0) ||
        (seconds > PUMP_MAX_SECONDS)
    )
    {
        sprintf(buff, "BAD_TIME");
        NRF_Set_ACK_Text(buff);
        return;
    }

    // ---------------------------------------------------------
    // Same session/id received again
    // ---------------------------------------------------------
    if(is_same_message(session_id, message_id))
    {
        if(
            (last_command_type == APP_CMD_RUN) &&
            (last_duration_seconds == seconds)
        )
        {
            // IMPORTANT:
            // Do not call pump_start_seconds() again.
            // A duplicate command must not extend irrigation.
            sprintf(
                buff,
                "DUP,%u,%u",
                session_id,
                message_id
            );

            NRF_Set_ACK_Text(buff);
        }
        else
        {
            sprintf(buff, "ID_CONFLICT");
            NRF_Set_ACK_Text(buff);
        }

        return;
    }

    // Queue response for the following POLL packet
    sprintf(
        buff,
        "ACK,RUN,%u,%u,%u",
        session_id,
        message_id,
        seconds
    );

    NRF_Set_ACK_Text(buff);

    remember_command(
        session_id,
        message_id,
        APP_CMD_RUN,
        seconds
    );

    // Local timed actuation
    pump_start_seconds(seconds);
}


//==============================================================
// OFF COMMAND
//
// OFF,<session>,<id>
// Example:
//   OFF,4711,13
//==============================================================

void handle_off_command(char *cmd)
{
    char buff[NRF_MAX_PAYLOAD + 1];

    unsigned char index = 4;

    unsigned int session_id;
    unsigned int message_id;

    session_id = parse_uint(cmd, &index);

    if(cmd[index] != ',')
    {
        sprintf(buff, "BAD_CMD");
        NRF_Set_ACK_Text(buff);
        return;
    }

    index++;

    message_id = parse_uint(cmd, &index);

    if(cmd[index] != 0)
    {
        sprintf(buff, "BAD_CMD");
        NRF_Set_ACK_Text(buff);
        return;
    }

    if(is_same_message(session_id, message_id))
    {
        if(last_command_type == APP_CMD_OFF)
        {
            sprintf(
                buff,
                "DUP,%u,%u",
                session_id,
                message_id
            );

            NRF_Set_ACK_Text(buff);
        }
        else
        {
            sprintf(buff, "ID_CONFLICT");
            NRF_Set_ACK_Text(buff);
        }

        return;
    }

    sprintf(
        buff,
        "ACK,OFF,%u,%u",
        session_id,
        message_id
    );

    NRF_Set_ACK_Text(buff);

    remember_command(
        session_id,
        message_id,
        APP_CMD_OFF,
        0
    );

    pump_stop();
}


//==============================================================
// PROTOCOL
//==============================================================

void protocol_prepare_response(char *cmd)
{
    char buff[NRF_MAX_PAYLOAD + 1];

    // PING
    if(command_equal(cmd, 1))
    {
        sprintf(buff, "PONG");
        NRF_Set_ACK_Text(buff);
        return;
    }

    // STATUS
    if(command_equal(cmd, 2))
    {
        prepare_status_response();
        return;
    }

    // POLL
    //
    // Do not overwrite the ACK payload already queued by the
    // preceding command.
    if(command_equal(cmd, 3))
    {
        return;
    }

    // RUN
    if(command_starts_run(cmd))
    {
        handle_run_command(cmd);
        return;
    }

    // OFF
    if(command_starts_off(cmd))
    {
        handle_off_command(cmd);
        return;
    }

    sprintf(buff, "UNKNOWN");
    NRF_Set_ACK_Text(buff);
}


//==============================================================
// NRF SERVICE
//==============================================================

void nrf_service(void)
{
    char rx_data[NRF_MAX_PAYLOAD + 1];
    unsigned char width;

    if(!NRF_DataReady())
    {
        return;
    }

    width = NRF_Read_Text(rx_data);

    if(width == 0)
    {
        return;
    }

    protocol_prepare_response(rx_data);
}


//==============================================================
// MAIN
//==============================================================

void main(void)
{
    init_system();

    // Fail-safe: always OFF before radio initialization
    pump_stop();

    NRF_Init();

    #asm("sei")

    while(1)
    {
        // Radio service
        nrf_service();

        // Independent local pump timeout
        pump_task();
    }
}
