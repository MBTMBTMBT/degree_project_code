import sys
import os
from glob import glob
from PySide2 import QtWidgets,QtCore,QtGui
from PySide2.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PySide2.QtCore import QDir, QTimer,Slot,QCoreApplication
from PySide2.QtGui import QPixmap,QImage
from ui_mainwindow import Ui_MainWindow
import cv2
import myframe
import time
from qtmodern import styles
from qtmodern import windows
import qtmodern

import sys, os
import PySide2

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
print(plugin_path)

# Initial global parameters

ActionCOUNTER = 0

# Fatigue model: Perclos model
# perclos = (Rolleye/Roll) + (Rollmouth/Roll)*0.2
Roll = 0 # the loop counter 
Rolleye = 0 # Blinking counter 
Rollmouth = 0 # Yawning counter 
Rollnoeye = 0 # No eye counter
Time = 0


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # Open the file type for class definition
        # initial type = 0 means user need to trigger opening action firstly
        self.f_type = 0

    def window_init(self):
        # set initial value of labels
        self.label.setText("Please open the camera")
        self.label_2.setText("Fatigue:")
        self.label_3.setText("Blinking: 0")
        self.label_4.setText("Yawning: 0")

        self.label_5.setText("Motion:")
        self.label_6.setText("Seatbelt:")
        self.label_7.setText("Mask:")

        self.label_8.setText("yes")
        self.label_11.setText("no")
        self.label_9.setText(" ")
        self.label_10.setText("")
        
        self.menu.setTitle("Start")
        self.open.setText("Open the camera")
        # Menu button slots connect to functions
        # not carema function but trigger defined CamConfig class which open carema video by cv2 
        self.open.triggered.connect(CamConfig_init)
        # Auto-adapt window playback
        self.label.setScaledContents(True)


# Carema class
class CamConfig:
    def __init__(self):
        Ui_MainWindow.printf(window,"The computer is turning on its camera...")
        # set up a timer
        self.v_timer = QTimer()
        # open carema by cv2
        self.cap = cv2.VideoCapture(0)
        # Failed to open carema
        if not self.cap:
            Ui_MainWindow.printf(window,"Failed to open camera")
            return
        # Set the timer period in milliseconds
        self.v_timer.start(20)
        # Connect timer cycle overflow slot function used to display a frame of video
        self.v_timer.timeout.connect(self.show_pic)
        # output the marked message in UI
        Ui_MainWindow.printf(window,"Load video successfully and start dangerous behaviour detection...")
        Ui_MainWindow.printf(window,"")
        
        
    def show_pic(self):
        # start the timer to remind the start time
        tstart = time.time()
        
        # Introduce defined global variables into a function
        global ActionCOUNTER,Roll,Rolleye,Rollmouth,Rollnoeye,Time,f
        
        # read one frame from carema
        success, frame = self.cap.read()
        
        if success:
            # Detector
            # Pass the frame read by the camera to the detection function myframe.frametest()
            head_class, sunglasses, mask, ret,frame = myframe.frametest(frame,Roll)
            TOTAL,mTOTAL,COUNTER,mCOUNTER,NO_EYE_COUNTER = ret

            # window.label_7.setText("Mask: " + mask)

            window.label_8.setText("no")
            window.label_11.setText("no" if mask == 'not_wearing' else 'yes')
            window.label_9.setText(head_class)
            #window.label_10.setText("  ")


            # Ret and frame, returned for the function
            # Ret is the detection result, ret is in the format of a lot of counter

            # Frame refers to the frame marked with the identification result, 
            # and draw the identification frame by cv2 on the video area

            window.label_3.setText("Blinking: " + str(TOTAL))
            window.label_4.setText("Yawning: " + str(mTOTAL))

            Rolleye += COUNTER
            Rollmouth += mCOUNTER
            #print(mCOUNTER)
            
            if NO_EYE_COUNTER != 0:
                Rollnoeye += 1

            # Show this frame in UI
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            window.label.setPixmap(QPixmap.fromImage(showImage))

            # Fatigue model
            # use defined time as one loop
            # Roll is the loop counter and +1 every loop
            tend = time.time()
            Time = Time + (tend-tstart) 
            Roll += 1

            # If time 
            # Produce one detection of fatigue
            if Time > 5:
                #print(Roll)
                # If the eyes are not found in more than half of the frames
                # Display one warning message on the front UI
                # and stop the detection
                if Rollnoeye > Roll/2:
                    window.label_10.setText("<font color=red>Unknown！！！</font>")
                    Ui_MainWindow.printf(window,"Can't detect your face, please adjust camera position or look forward")

                else:
                    #print(Rolleye)
                    #print(Roll)
                    # Calcueate the average Perclos model score in defined time
                    perclos = (Rolleye/Roll)*0.9 + (Rollmouth/Roll)*0.1
                    # Print the score of Perclos score
                    Ui_MainWindow.printf(window,"In the past defined time, the score of Perclos model is "+str(round(perclos,3)))
                    # When the Perclos model score exceeds 0.38 in the defined time
                    # we determined it is in fatigue statement
                    if perclos > 0.38:
                        Ui_MainWindow.printf(window,"The current state is fatigue")
                        window.label_10.setText("<font color=red>Fatigue！！！</font>")
                        os.system("beep.mp4")
                    else:
                        Ui_MainWindow.printf(window,"The current state is awake")
                        window.label_10.setText("Awake")
                        # set a new loop for fatigue detection
                    Ui_MainWindow.printf(window,"Start a new round of monitoring...")
                
                Ui_MainWindow.printf(window,"")# for clear feedback to user     
                # 
                # Reset the counter to zero
                Roll = 0
                Rolleye = 0
                Rollmouth = 0
                Rollnoeye = 0
                Time = 0   
        
def CamConfig_init():
    # the triggered action: detector and video shower
    window.f_type = CamConfig()


if __name__ == '__main__':
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # Low-level administrative functions such as handling of initialization entry parameters
    app = QtWidgets.QApplication(sys.argv) 
    window = MainWindow() 
    window.window_init() # initial label
    window.setWindowTitle("Dangerous driving behaviour")
    qtmodern.styles.dark(app)
    mwindow = qtmodern.windows.ModernWindow(window)
    mwindow.show() # show window
    sys.exit(app.exec_()) 
