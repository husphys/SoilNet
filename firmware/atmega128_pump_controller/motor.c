//==============================================================
// motor.c - low-level, verified mechanical parameters
//==============================================================

void control_motor(unsigned char motor,unsigned char dir_motor,unsigned char speed);
void IO_Init(void);
void motor1_up(void); void motor1_down(void); void motor1_stop(void);
void motor2_left(void); void motor2_right(void); void motor2_stop(void);
void pump_on(void); void pump_off(void);
void relay_on(void); void relay_off(void);
void all_motor_stop(void);
unsigned char tc1_pressed(void); unsigned char tc2_pressed(void);
unsigned char tc3_pressed(void); unsigned char tc4_pressed(void);
unsigned char water_full_detected(void);

void IO_Init(void)
{
    DDRC.0=0; DDRC.1=0; DDRC.2=0; DDRC.3=0; DDRC.4=0;
    PORTC.0=1; PORTC.1=1; PORTC.2=1; PORTC.3=1; PORTC.4=1;

    DDRC.7=0; PORTC.7=1;
    DDRA.7=0; DDRA.6=0;
    PORTA.7=1; PORTA.6=1;
    delay_ms(10);
}

void control_motor(unsigned char motor,unsigned char dir_motor,unsigned char speed)
{
    switch(motor)
    {
        case 1:
            if(dir_motor==0){DIR_1=dir_motor;PWM_1=speed;}
            else{DIR_1=dir_motor;PWM_1=255-speed;}
            break;

        case 2:
            if(dir_motor==0){DIR_2=dir_motor;PWM_2=speed;}
            else{DIR_2=dir_motor;PWM_2=255-speed;}
            break;

        case 3:
            if(dir_motor==0){DIR_3=dir_motor;PWM_3=speed;}
            else{DIR_3=dir_motor;PWM_3=255-speed;}
            break;

        case 4:
            if(dir_motor==1){DIR_4=dir_motor;PWM_4=speed;}
            else{DIR_4=dir_motor;PWM_4=255-speed;}
            break;
    }
}

// EXACT values verified on hardware
void motor1_up(void){control_motor(motor_1,M1_DIR_UP,80);}
void motor1_down(void){control_motor(motor_1,M1_DIR_DOWN,38);}
void motor1_stop(void){control_motor(motor_1,M1_DIR_UP,0);}

void motor2_left(void){control_motor(motor_2,M2_DIR_LEFT,100);}
void motor2_right(void){control_motor(motor_2,M2_DIR_RIGHT,100);}
void motor2_stop(void){control_motor(motor_2,M2_DIR_RIGHT,0);}

void pump_on(void){control_motor(motor_3,M3_DIR_PUMP,SPEED_PUMP);}
void pump_off(void){control_motor(motor_3,M3_DIR_PUMP,0);}

// Keep current relay setting unchanged
void relay_on(void){control_motor(motor_4,RELAY_DIR,200);}
void relay_off(void){control_motor(motor_4,RELAY_DIR,0);}

void all_motor_stop(void){motor1_stop();motor2_stop();pump_off();}

unsigned char tc1_pressed(void){if(TC_1==TC_ACTIVE)return 1;return 0;}
unsigned char tc2_pressed(void){if(TC_2==TC_ACTIVE)return 1;return 0;}
unsigned char tc3_pressed(void){if(TC_3==TC_ACTIVE)return 1;return 0;}
unsigned char tc4_pressed(void){if(TC_4==TC_ACTIVE)return 1;return 0;}
unsigned char water_full_detected(void){if(WATER_FULL_SENSOR==WATER_FULL_ACTIVE)return 1;return 0;}
