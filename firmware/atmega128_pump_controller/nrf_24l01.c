//==============================================================
// nrf_24l01.c
// nRF24L01 ACK Payload
// ATmega128 permanent PRX
// Compiler: CodeVisionAVR
//
// Pi:
//   permanent PTX
//
// AVR:
//   permanent PRX
//
// RF:
//   Address = A1 A1 A1 A1 A1
//   Channel = 2
//   1 Mbps
//   CRC16
//   Dynamic Payload ON
//   ACK Payload ON
//
// CHU Y CODEVISION:
// Tat ca ACK text dua vao ham nay deu nam trong RAM.
// Chuoi co dinh se duoc tao truoc bang sprintf(buff,"...")
// trong main.c hoac NRF_Init().
//==============================================================


#define NRF_R_REGISTER          0x00
#define NRF_W_REGISTER          0x20
#define NRF_R_RX_PAYLOAD        0x61
#define NRF_FLUSH_TX            0xE1
#define NRF_FLUSH_RX            0xE2
#define NRF_ACTIVATE            0x50
#define NRF_W_ACK_PAYLOAD_P0    0xA8

#define NRF_CONFIG              0x00
#define NRF_EN_AA               0x01
#define NRF_EN_RXADDR           0x02
#define NRF_SETUP_AW            0x03
#define NRF_SETUP_RETR          0x04
#define NRF_RF_CH               0x05
#define NRF_RF_SETUP            0x06
#define NRF_STATUS              0x07
#define NRF_RX_ADDR_P0          0x0A
#define NRF_DYNPD               0x1C
#define NRF_FEATURE             0x1D

#define NRF_ADDRESS_BYTE        0xA1


//==============================================================
// SOFTWARE SPI
//==============================================================

unsigned char SPI_RW(unsigned char Buff)
{
    unsigned char bit_ctr;


    for(bit_ctr=0;bit_ctr<8;bit_ctr++)
    {
        if(Buff & 0x80)
            MOSI=1;
        else
            MOSI=0;


        delay_us(5);


        Buff <<= 1;


        SCK=1;

        delay_us(5);


        if(MISO)
            Buff |= 0x01;


        SCK=0;
    }


    return Buff;
}


//==============================================================
// REGISTER ACCESS
//==============================================================

void NRF_Write(
    unsigned char reg,
    unsigned char value
)
{
    CSN=0;

    SPI_RW(
        NRF_W_REGISTER |
        (reg & 0x1F)
    );

    SPI_RW(value);

    CSN=1;

    delay_us(10);
}


unsigned char NRF_Read(
    unsigned char reg
)
{
    unsigned char value;


    CSN=0;

    SPI_RW(
        NRF_R_REGISTER |
        (reg & 0x1F)
    );

    value=
        SPI_RW(0xFF);

    CSN=1;


    return value;
}


void NRF_Command(
    unsigned char command
)
{
    CSN=0;

    SPI_RW(command);

    CSN=1;

    delay_us(10);
}


void NRF_Write_Address(
    unsigned char reg,
    unsigned char value
)
{
    unsigned char i;


    CSN=0;

    SPI_RW(
        NRF_W_REGISTER |
        (reg & 0x1F)
    );


    for(i=0;i<5;i++)
    {
        SPI_RW(value);
    }


    CSN=1;

    delay_us(10);
}


//==============================================================
// FEATURE / FIFO
//==============================================================

void NRF_Activate_Features(void)
{
    CSN=0;

    SPI_RW(NRF_ACTIVATE);
    SPI_RW(0x73);

    CSN=1;

    delay_us(20);
}


void NRF_Clear_IRQ(void)
{
    NRF_Write(
        NRF_STATUS,
        0x70
    );
}


void NRF_Flush_RX(void)
{
    NRF_Command(
        NRF_FLUSH_RX
    );
}


void NRF_Flush_TX(void)
{
    NRF_Command(
        NRF_FLUSH_TX
    );
}


//==============================================================
// DYNAMIC RX PAYLOAD
//==============================================================

unsigned char NRF_Read_Payload_Width(void)
{
    unsigned char width;


    CSN=0;

    SPI_RW(0x60);

    width=
        SPI_RW(0xFF);

    CSN=1;


    return width;
}


unsigned char NRF_Read_Text(
    char *data
)
{
    unsigned char i;
    unsigned char width;


    CE=0;


    width=
        NRF_Read_Payload_Width();


    if(
        (width==0) ||
        (width>NRF_MAX_PAYLOAD)
    )
    {
        NRF_Flush_RX();

        NRF_Clear_IRQ();

        CE=1;

        return 0;
    }


    CSN=0;

    SPI_RW(
        NRF_R_RX_PAYLOAD
    );


    for(i=0;i<width;i++)
    {
        data[i]=
            SPI_RW(0xFF);
    }


    CSN=1;


    data[width]=0;


    NRF_Write(
        NRF_STATUS,
        0x40
    );


    CE=1;


    return width;
}


//==============================================================
// ACK PAYLOAD
//
// text PHAI la chuoi trong RAM.
//
// Vi du:
//   char buff[20];
//   sprintf(buff,"PONG");
//   NRF_Set_ACK_Text(buff);
//==============================================================

void NRF_Set_ACK_Text(
    char *text
)
{
    unsigned char i;
    unsigned char len=0;


    while(
        (text[len]!=0) &&
        (len<NRF_MAX_PAYLOAD)
    )
    {
        len++;
    }


    CE=0;


    // Chi giu response moi nhat
    NRF_Flush_TX();


    if(len>0)
    {
        CSN=0;


        SPI_RW(
            NRF_W_ACK_PAYLOAD_P0
        );


        for(i=0;i<len;i++)
        {
            SPI_RW(
                text[i]
            );
        }


        CSN=1;
    }


    CE=1;
}


//==============================================================
// INIT nRF
//==============================================================

void NRF_Init(void)
{
    unsigned char feature;

    char buff[20];


    CE=0;
    CSN=1;
    SCK=0;
    MOSI=0;


    delay_ms(100);


    // CRC16, power down
    NRF_Write(
        NRF_CONFIG,
        0x0C
    );


    // Auto ACK pipe 0
    NRF_Write(
        NRF_EN_AA,
        0x01
    );


    // Enable RX pipe 0
    NRF_Write(
        NRF_EN_RXADDR,
        0x01
    );


    // Address width = 5
    NRF_Write(
        NRF_SETUP_AW,
        0x03
    );


    // Retry
    NRF_Write(
        NRF_SETUP_RETR,
        0x5F
    );


    // Channel 2
    NRF_Write(
        NRF_RF_CH,
        0x02
    );


    // 1 Mbps
    NRF_Write(
        NRF_RF_SETUP,
        0x02
    );


    // RX address
    NRF_Write_Address(
        NRF_RX_ADDR_P0,
        NRF_ADDRESS_BYTE
    );


    // EN_DPL + EN_ACK_PAY
    NRF_Write(
        NRF_FEATURE,
        0x06
    );


    feature=
        NRF_Read(
            NRF_FEATURE
        );


    // Mot so module clone can ACTIVATE 0x73
    if(
        (feature & 0x06) != 0x06
    )
    {
        NRF_Activate_Features();


        NRF_Write(
            NRF_FEATURE,
            0x06
        );
    }


    // Dynamic payload pipe 0
    NRF_Write(
        NRF_DYNPD,
        0x01
    );


    NRF_Flush_RX();
    NRF_Flush_TX();
    NRF_Clear_IRQ();


    // Power up + PRX + CRC16
    NRF_Write(
        NRF_CONFIG,
        0x0F
    );


    delay_ms(2);


    CE=1;


    // ACK payload dau tien
    sprintf(
        buff,
        "READY"
    );


    NRF_Set_ACK_Text(
        buff
    );
}


//==============================================================
// RX READY
//==============================================================

unsigned char NRF_DataReady(void)
{
    if(
        NRF_Read(NRF_STATUS) &
        0x40
    )
    {
        return 1;
    }


    return 0;
}
