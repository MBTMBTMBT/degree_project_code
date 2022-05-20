#!/usr/bin/python
# coding=UTF-8
import RPi.GPIO as GPIO
import serial
import time
from threading import Timer
from EC200U import*      
ec200u=EC200U('/dev/ttyS0',115200) # open relevant port
ec200u.ATSignle('ATE0\r\n','OK')# judge whether module exists
ec200u.ATSignle('AT+QGPS=1\r\n','OK')# open GNSS, set gps power on
time.sleep(0.5)# the unit is second
while True:
    ec200u.ATSignle('AT+QGPSGNMEA=\"rmc\"\r\n','OK')# get gps data
    print(ec200u.rxData)
    if ec200u.rxData.find('$GNRMC')!=-1:#
        line = str(ec200u.rxData).split(',')  # let line separated by “，”
        if line[4]=='N':# location successfully
            weidu = float(line[3][:2]) + float(line[3][2:])/60
            # Read the fifth string information, from 0 to 3 to the longitude,
            # followed by a string of division 60 to convert the minutes to degrees
            jingdu = float(line[5][:3]) + float(line[5][3:])/60
            # the same as above
            print("longitude:",'{:.6f}'.format(jingdu))
            print("latitude:",'{:.6f}'.format(weidu))
    time.sleep(0.5)# the unit is second

         
         