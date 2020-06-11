# Arduino_RC_Car_Python_OpenCV_Gesture_Control

## Demo of the project below

![Project Demo](Demo/ezgif-5-bc83da8bc06a.gif)

## Description

Here i used a ESP8266 Arduino module in the RC car to act as server and also the WIFI access point. The code for the ESP6266 module is located in the Arduino Code module which you need to flash into your ESP module.

Then you can run the python code which acts as a client and your laptops wifi must be connected to the ESP8266 modules Access point.Next you can run the python code which had been trained on a ROCK, PAPER, Scissors Dataset and it used Post request to communicate to the ESP module based on the arguments recieved in the post request the esp module is configues to send the power to move either Forward, Backward or Right.
