void manual_mode(){
unsigned char buff[10];    
    glcd_clear();
    glcd_moveto(3,10);
    glcd_outtext("Manual Mode"); 
    while(1){ 
        RX_Mode();
        if(IRQ == 0){
            BL_Nokia=~BL_Nokia; 
            LED=~LED;
            xx=RF_RX_Read();
            sprintf(buff, " %u ", xx);
            glcd_outtextxy(36,25,buff);

                //---------analog left----------
                if(xx==1) {forward(245,245,245,245);}        //a-trai-thang   
                if(xx==2) {backward(245,245,245,245);}          //a-trai-lui
                if(xx==4) {turn_left(230,230,230,230);}         //a-trai-trai                
                if(xx==3) {turn_right(230,230,230,230);}         //a-trai-phai 
              
                //---------digital left----------  
                if(xx==5)  {forward(245,245,245,245);}        //a-trai-thang   
                if(xx==6) {backward(245,245,245,245);}          //a-trai-lui
                if(xx==7) {left(230,230,230,230);}
                if(xx==8) {right(230,230,230,230);}
                
           if(xx==9)  {
                    low_arm();
                    updown_state = 1;}         //up                
                
                if(xx==10) {
                    low_arm(); 
                    updown_state = -1;}          //down
               
           
               if(xx==13){
                   hand_up();}                              
            
               if(xx==14){
                   hand_down();}            
            
             
            
            
             if(xx==12) {    //left  
           
                    open_box(); }  
                     
             if(xx==11) {    //left  
                    
                    open_close_box();
                    delay_ms(200);
                    low_arm(); 
                    updown_state = -1;
                     }
           
              
              if(xx==15)                             //tron
                    {        
                      open_claw(); 
                    } 
                if(xx==16)                             // vuong   
                    {
                      close_claw();
                    }
                if(xx==19)                             // L1   
                    {
                    drop_recycl();    
                    }

                if(xx==21)                             // R1   
                    {
                        drop_tree();
                    }
                if(xx==22)                             // R2   
                    {
                        drop_tree();
                    }
                if (xx == 20 ) // reset state
                    {
                     drop_recycl();
                    }
                if(xx==17)                             // Select   
                    {
                      drop_ball();
                    }                        

                if(xx==0) 
                    {   
                        stop();
                        box_stop(); 
                    }
            RX_Config();
        }


    }
  }