int incomingByte;
const int LED=12;
int onTime;
int tmp;
#include <Servo.h>
Servo myServo;
Servo greenLaserServo;
Servo blueLaserServo;

int angle;
int greenLaserAngle;
int blueLaserAngle;

void setup() {
  myServo.attach(9);                       // Connect both servos.
  greenLaserServo.attach(10);
  blueLaserServo.attach(5);
  Serial.begin(9600);                      //initialize serial COM at 9600 baudrate
  pinMode(LED, OUTPUT);                    //declare the LED pin (12) as output
  digitalWrite(LED, LOW);
  delay(100);
  Serial.println("Hello!,How are you Python ?");
  onTime = 1000;
  delay(100);
  angle = 90;

  myServo.write(angle);
  greenLaserAngle = 90;
  greenLaserServo.write(greenLaserAngle);

  blueLaserAngle = 90;
  blueLaserServo.write(blueLaserAngle);
}

void loop() {
  // put your main code here, to run repeatedly:
  // TODO rewrite this code to make it more readable and easier to use.
  // See if there is a case statement which may be used. Perhaps  2 letter signals?
  if (Serial.available() > 0) {
    // read the oldest byte in the serial buffer:
    incomingByte = Serial.read();
    // if it's a capital H (ASCII 72), turn on the LED:
    if (incomingByte == 'H') {
      digitalWrite(LED, HIGH);
      Serial.println("Turning on");
    }
    // if it's an L (ASCII 76) turn off the LED:
    if (incomingByte == 'L') {
      digitalWrite(LED, LOW);
      Serial.println("Turning off");
    }
    if (incomingByte == 'T') {
      digitalWrite(LED, HIGH);
      Serial.println("Turning on then off");
      delay(onTime);
      digitalWrite(LED, LOW);
      //Serial.println("Turning off");
    }
    if (incomingByte == 'S') {
      Serial.println("S recieved expecting int next");
      while (Serial.available()==0){delay(10);}
      
      tmp = Serial.parseInt();
      if (tmp > 0){
        Serial.println("On time set to: ");
        Serial.println(tmp);
        onTime = tmp;
      }
      else{
        Serial.println("Trying to set invalid timeframe!");
        Serial.println(tmp);
      }    
    }
   if (incomingByte == 'O') {
    Serial.println("Opening the shutter.");
    angle = 180;
    myServo.write(angle);       
    }
   if (incomingByte == 'C') {
    Serial.println("Closing the shutter.");
    angle = 90;
    myServo.write(angle);      
   }
   if (incomingByte == 'G'){
    Serial.println("Unblocking green laser.");
    greenLaserAngle = 180;
    greenLaserServo.write(greenLaserAngle);
   }
   if (incomingByte == 'B'){
    Serial.println("Blocking green laser.");
    greenLaserAngle = 90;
    greenLaserServo.write(greenLaserAngle);
   }
   if (incomingByte == 'Q'){
    Serial.println("Unblocking blue laser.");
    blueLaserAngle = 120;
    blueLaserServo.write(blueLaserAngle);
   }
   if (incomingByte == 'W'){
    Serial.println("Blocking blue laser.");
    blueLaserAngle = 90;
    blueLaserServo.write(blueLaserAngle);
   }
  }
}
