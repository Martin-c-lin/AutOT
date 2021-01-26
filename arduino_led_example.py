import serial, time
from random import randint
ArduinoUnoSerial = serial.Serial('com4',9600) # may need to adapt this

#print(ArduinoUnoSerial.readline())

print ("You have new message from Arduino")
running = True

commands = ['H','L','T','1','2','3']

while running:         #Do this forever
    #var = randint(0,1) #raw_input()                                          #get input from user
    #var = str(var)
    var = input("Please enter a string:\n")
    print(var)
    #ArduinoUnoSerial.write(1)
    if var in commands:                                                #if the value is 1
        message = var.encode('utf-8')
        ArduinoUnoSerial.write(message)                      #send 1 to the arduino's Data code
        print ("LED turned ON")
        time.sleep(0.1)
    # if (var == 'L'):
    #     ArduinoUnoSerial.write(b'L')            #send 0 to the arduino's Data code
    #     print ("LED turned OFF")
    #     time.sleep(0.1)
    if (var == 'quit'): #if the answer is (fine and you)
        ArduinoUnoSerial.write(b'L')            #send 0 to the arduino's Data code
        ArduinoUnoSerial.close()
        running = False
