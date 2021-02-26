#include <Servo.h>
Servo myServo;
int const potPin = A0;
int potVal;
int angle;

void setup() {
 myServo.attach(9);
 Serial.begin(9600);
}
void loop() {
 //potVal = analogRead(potPin);
 if (Serial.available() > 0) {
  potVal = Serial.parseInt();
   Serial.print("potVal: ");
   Serial.print(potVal);
   angle = map(potVal, 0, 10, 0, 179);
   Serial.print(", angle: ");
   Serial.println(angle);
   myServo.write(angle);
   delay(3000);
 }

 delay(15);
}
