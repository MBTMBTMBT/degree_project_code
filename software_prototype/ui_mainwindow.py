# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################


from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1100, 600)
        # this is the action of opening carema action
        # not cause by qt but a triggered action connected to cv2
        # just showing as a triggered action here called "open"
        self.open = QAction(MainWindow)
        self.open.setObjectName(u"open")
        # get the central wide
        self.GetCentralWidth = QWidget(MainWindow)
        self.GetCentralWidth.setObjectName(u"GetCentralWidth")
        # the area prepared for the video
        # Label1 is the marked words for opening carema
        self.verticalLayout_2 = QVBoxLayout(self.GetCentralWidth)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.GetCentralWidth)
        self.label.setObjectName(u"label")
        self.label.setMinimumSize(QSize(720, 480))
        self.label.setMaximumSize(QSize(720, 480))

        self.horizontalLayout.addWidget(self.label)

        # label 2-11 for counter and detector
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_2 = QLabel(self.GetCentralWidth)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMaximumSize(QSize(120, 30))

        self.horizontalLayout_5.addWidget(self.label_2)

        self.label_10 = QLabel(self.GetCentralWidth)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setMaximumSize(QSize(180, 30))

        self.horizontalLayout_5.addWidget(self.label_10)

        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_3 = QLabel(self.GetCentralWidth)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMaximumSize(QSize(150, 30))

        self.horizontalLayout_2.addWidget(self.label_3)

        self.label_4 = QLabel(self.GetCentralWidth)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMaximumSize(QSize(150, 30))

        self.horizontalLayout_2.addWidget(self.label_4)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_5 = QLabel(self.GetCentralWidth)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMaximumSize(QSize(120, 30))

        self.horizontalLayout_4.addWidget(self.label_5)

        self.label_9 = QLabel(self.GetCentralWidth)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setMaximumSize(QSize(180, 30))

        self.horizontalLayout_4.addWidget(self.label_9)

        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_6 = QLabel(self.GetCentralWidth)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setMaximumSize(QSize(80, 30))

        self.horizontalLayout_3.addWidget(self.label_6)

        self.label_8 = QLabel(self.GetCentralWidth)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setMaximumSize(QSize(70, 30))

        self.horizontalLayout_3.addWidget(self.label_8)


        self.label_7 = QLabel(self.GetCentralWidth)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setMaximumSize(QSize(60, 30))

        self.horizontalLayout_3.addWidget(self.label_7)

        self.label_11 = QLabel(self.GetCentralWidth)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setMaximumSize(QSize(90, 30))

        self.horizontalLayout_3.addWidget(self.label_11)

        self.verticalLayout.addLayout(self.horizontalLayout_3)

        # part for text browser
        # the real time statement of detection refresh continously
        self.textBrowser = QTextBrowser(self.GetCentralWidth)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setMaximumSize(QSize(300, 360))

        self.verticalLayout.addWidget(self.textBrowser)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        # Set menu windows
        MainWindow.setCentralWidget(self.GetCentralWidth)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1060, 26))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # add open action to the menu botton
        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.open)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.open.setText(QCoreApplication.translate("MainWindow", u"Open camera", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"Open", None))
    # retranslateUi

    # text browser function
    def printf(self, mes):
        self.textBrowser.append(mes)  # show the real time meessage in the designated area
        self.cursot = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursot.End)
