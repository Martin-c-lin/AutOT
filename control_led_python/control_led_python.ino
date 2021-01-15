int incomingByte;
const int LED=12;
void setup() { 
  Serial.begin(9600);                      //initialize serial COM at 9600 baudrate
  pinMode(LED, OUTPUT);                    //declare the LED pin (12) as output
  digitalWrite(LED, LOW);
  delay(100);
  Serial.println("Hello!,How are you Python ?");
}

void loop() {
  // put your main code here, to run repeatedly:
  
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
  }
}
