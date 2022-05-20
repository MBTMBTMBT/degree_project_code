# coding=UTF-8
import RPi.GPIO as GPIO
import serial
import time
class  EC200U:
    def __init__(self,port,baudrate):
        self.uart0 = serial.Serial(port,baudrate)#
        if self.uart0.isOpen == True:
            self.uart0.open()
            self.uart0.flushInput()
    def ATinit(self,ATcmd,OKcmd):# Send AT instruction to control module
         self.uart0.write(ATcmd.encode())
         #self.uart0.write(b'\r\n')
         time.sleep(0.3)# unit is second
         self.rxData=''
         self.rxData = self.uart0.read(self.uart0.inWaiting())
         print(self.rxData.decode())
         if OKcmd not in  self.rxData.decode():# The returned data is not in, indicating that the read is abnormal
              return 0
         else:#读取正确
              return 1   
    def ATSignle(self,ATcmd,OKcmd):# AT instructions are sent to the module, but only once AT a time, and the user connects to the server
         self.uart0.write(ATcmd.encode())
         time.sleep(2)# unit is second
         self.rxData=''
         self.rxData = self.uart0.read(self.uart0.inWaiting())# read the content
         print(self.rxData.decode())
         if OKcmd not in  self.rxData.decode():# The returned data is not in, indicating that the read is abnormal
               return 0
         else:#  read is correct
               return 1
    def TCPSend(self,ATcmd,OKcmd):# Sends TCP data to the server
         self.uart0.write(ATcmd.encode())
         time.sleep(0.5)# unit is second
         #self.rxData=''
         self.rxData = self.uart0.read(self.uart0.inWaiting())# read the content
         print(self.rxData.decode())
         if OKcmd not in  self.rxData.decode():# The returned data is not in, indicating that the read is abnormal
               return 0
         else:# read is correct
               return 1
    def TCPREC(self):#recieve server data
        #self.rxData=''
        self.rxData = self.uart0.read(self.uart0.inWaiting())# read the content
        print(self.recData.decode())
    def Writedata(self,data):
        self.uart0.write(data)
        
    
        
