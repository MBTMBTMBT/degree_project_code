#!/usr/bin/python
# coding=UTF-8
import RPi.GPIO as GPIO
import serial
import time
from threading import Timer
from EC200U import*      
rec_buff = ''
APN = 'CMNET'
ServerIP = '139.196.135.135'
Port = '1883'
username='GPS_test'
ProductKey='h27sAMyfKAm'
DeviceName='test_1'
DeviceSecret='c1cff0799f9611e6b4dbb151dfdf8ebf'
pubtopic='/sys/h27sAMyfKAm/test_1/thing/event/property/post'
wenshidu={'wendu':23,'shidu':42}
jingweidu={'jingdu':117.130638,'weidu':31.838612}
class EC200UCtr:# Send AT instruction to control module
    def EC200U_AT(self):
        while(ec200u.ATinit('ATE0\r\n','OK')==0):# Check whether the module exists
            pass
        while(ec200u.ATinit('AT+CIMI\r\n','460')==0):# Check whether the SIM card exists
            pass
        while(ec200u.ATinit('AT+CGATT?\r\n','+CGATT: 1')==0):# Check whether the network is successfully registered
            pass
        ec200u.ATSignle('AT+QGPS=1\r\n','OK')# set gps power on
        ec200u.ATSignle('AT+QMTDISC=0\r\n','OK')# For the last disconnection, not need to determine whether the connection is successfully closed
        ec200u.ATSignle('AT+QMTCFG=\"aliauth\",0,\"'+ProductKey+'\",\"'+DeviceName+'\",\"'+DeviceSecret+'\"\r\n','OK')
        if ec200u.ATSignle('AT+QMTOPEN=0,\"'+ServerIP+'\",'+Port+'\r\n','+QMTOPEN: 0,0')==1:# Connect to the MQTT server and confirm the connection status
            self.tcpconok=1# Succesfully connect to the server
        else:
            self.tcpconok=0#fail
        if self.tcpconok==1:
          if ec200u.ATSignle('AT+QMTCONN=0,\"'+username+'\"\r\n','+QMTCONN: 0,0,0')==1:
              self.mqttconok=1# mqtt is ok
          else:
              self.mqttconnok=0# mqtt is fail
    def EC200U_REC(self):
        if '+QIURC:' in ec200u.rxData:# rec data from server
            recvalue=ec200u.rxData.find('LED10')# LED1 OFF
            if recvalue!=-1:#
                GPIO.output(20,GPIO.LOW)
                print('LED10')
            recvalue=ec200u.rxData.find('LED11')# LED1 ON
            if recvalue!=-1:#
                GPIO.output(20,GPIO.HIGH)
                print('LED11')
            recvalue=ec200u.rxData.find('LED20')# LED2 OFF
            if recvalue!=-1:#
                GPIO.output(21,GPIO.LOW)
                print('LED20')
            recvalue=ec200u.rxData.find('LED21')#LED2 ON
            if recvalue!=-1:#
                GPIO.output(21,GPIO.HIGH)
                print('LED21')
        ec200u.rxData=''
ec200u=EC200U('/dev/ttyS0',115200)
ec200uctr=EC200UCtr()
ec200uctr.EC200U_AT()
GPIO.setmode(GPIO.BCM)
GPIO.setup(20,GPIO.OUT)#LED1
GPIO.setup(21,GPIO.OUT)#LED2
wenshidu['wendu']=23#key zidian
wenshidu['shidu']=77#key zidian
while True:
    if ec200uctr.mqttconok==1:#conn
        ec200u.ATSignle('AT+QGPSGNMEA=\"rmc\"\r\n','OK')# get gps data
        if ec200u.rxData.find('$GNRMC')!=-1:#
           line = str(ec200u.rxData).split(',')  # # let line separated by “，”
        if line[4]=='N':# location successfully
            latitude = float(line[3][:2]) + float(line[3][2:])/60
            # Read the fifth string information, from 0 to 3 to the longitude,
            # followed by a string of division 60 to convert the minutes to degrees
            longitude = float(line[5][:3]) + float(line[5][3:])/60
            # the same as above
            print("longitude:",'{:.6f}'.format(longitude))
            print("latitude:",'{:.6f}'.format(latitude))
            jingweidu['jingdu']='{:.6f}'.format(longitude)
            jingweidu['weidu']='{:.6f}'.format(latitude)
        Message='{\"id\":\"26\",\"version\":\"1.0\",\"params\":{\"CurrentTemperature\":{\"value\":'+str(wenshidu['wendu'])+'},"CurrentHumidity":{\"value\":'+str(wenshidu['shidu'])+'},\"GeoLocation\":{\"Latitude\":'+str(jingweidu['weidu'])+',\"Longitude\":'+str(jingweidu['jingdu'])+'}},\"method\":\"thing.event.property.post\"}'
        Messagelength=str(len(Message))#get data length
        ec200u.ATSignle('AT+QMTPUBEX=0,0,0,0,\"'+pubtopic+'\",'+Messagelength+'\r\n', '>')
        if ec200u.ATSignle(Message, '+QMTPUBEX: 0,0,0')==1:# data sends successfully
            print("data send is OK")
            time.sleep(1)# unit is second
            ec200uctr.EC200U_REC()
         
         