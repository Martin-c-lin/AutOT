import serial, time
from random import randint
ArduinoUnoSerial = serial.Serial('com3',9600) # may need to adapt this

#print(ArduinoUnoSerial.readline())

print ("You have new message from Arduino")
running = True

commands = ['H','L','T','1','2','3']

while running:
    var = input("Please enter a string:\n")
    var = 'quit' if len(var)<1 else var
    print(var[0])
    if var in commands:                                                #if the value is 1
        message = var.encode('utf-8')
        ArduinoUnoSerial.write(message)                      #send 1 to the arduino's Data code
        print ("LED turned ON")
        time.sleep(0.1)
    elif var[0] == 'S':
        print('O yeah')
        try:
            ArduinoUnoSerial.write(var.encode())
        except Exception as ex:
            print(f"ERROR: {ex}")
    print(ArduinoUnoSerial.readline())
    #print(ArduinoUnoSerial.readline())

    if (var == 'quit'): #if the answer is (fine and you)
        ArduinoUnoSerial.write(b'L')            #send 0 to the arduino's Data code
        ArduinoUnoSerial.close()
        running = False
