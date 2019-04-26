import sys
import csv
import numpy as np
import cv2 as cv
from PyQt5 import QtCore, QtWidgets, QtGui
from AS_System_Func import sysInit, inViewCheck, onlineSVM, childFlow, DomOriMag, oriJumpReduct, MagOriToMove, MoveComp, bottomSearch, leftSearch

loc0Name, loc1Name, loc2Name = "", "", ""

class Window(QtWidgets.QMainWindow):

    saveVideo, saveData = 0, 0
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(600, 100, 640, 650)
        self.setWindowTitle("Autism Screening System")
        self.setWindowIcon(QtGui.QIcon('Robotics.png'))
        
        self.Home()

    def Home(self):
        
        global lineEdit0, lineEdit1, lineEdit2, btn1, btn2
        btn0 = QtWidgets.QPushButton("Load Video", self)
        btn0.clicked.connect(self.load_location0)
        btn0.setGeometry(QtCore.QRect(400, 40, 130, 30))

        lineEdit0 = QtWidgets.QLineEdit(self)
        lineEdit0.setGeometry(QtCore.QRect(120, 40, 260, 30))

        label0 = QtWidgets.QLabel("Path:", self)
        label0.setGeometry(QtCore.QRect(60, 40, 50, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        label0.setFont(font)

        checkBox1 = QtWidgets.QCheckBox("Save Output Video", self)
        checkBox1.setGeometry(QtCore.QRect(60, 120, 130, 20))
        checkBox1.stateChanged.connect(self.state_save_video)

        btn1 = QtWidgets.QPushButton("Location", self)
        btn1.clicked.connect(self.load_location1)
        btn1.setGeometry(QtCore.QRect(450, 170, 130, 30))
        btn1.setEnabled(self.saveVideo)

        lineEdit1 = QtWidgets.QLineEdit(self)
        lineEdit1.setGeometry(QtCore.QRect(170, 170, 260, 30))
        lineEdit1.setEnabled(self.saveVideo)

        checkBox2 = QtWidgets.QCheckBox("Save Output Locations", self)
        checkBox2.setGeometry(QtCore.QRect(60, 250, 150, 20))
        checkBox2.stateChanged.connect(self.state_save_data)

        label1 = QtWidgets.QLabel("Path:", self)
        label1.setGeometry(QtCore.QRect(110, 170, 50, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        label1.setFont(font)

        btn2 = QtWidgets.QPushButton("Location", self)
        btn2.clicked.connect(self.load_location2)
        btn2.setGeometry(QtCore.QRect(450, 300, 130, 30))
        btn2.setEnabled(self.saveData)

        lineEdit2 = QtWidgets.QLineEdit(self)
        lineEdit2.setGeometry(QtCore.QRect(170, 300, 260, 30))
        lineEdit2.setEnabled(self.saveData)

        label2 = QtWidgets.QLabel("Path:", self)
        label2.setGeometry(QtCore.QRect(110, 300, 50, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        label2.setFont(font)

        btn3 = QtWidgets.QPushButton("Start", self)
        btn3.clicked.connect(self.Autism_Screening)
        btn3.setGeometry(QtCore.QRect(400, 430, 130, 130))
        font = QtGui.QFont()
        font.setPointSize(20)
        btn3.setFont(font)

        label3 = QtWidgets.QLabel("CASS", self)
        label3.setGeometry(QtCore.QRect(60, 430, 310, 130))
        font = QtGui.QFont()
        font.setFamily("Parchment")
        font.setPointSize(60)
        font.setItalic(False)
        label3.setFont(font)
        
        self.show()

    def load_location0(self):
        global loc0Name
        loc0Name, s = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Video')
        lineEdit0.insert(loc0Name) # Defined in Home

    def load_location1(self):
        global loc1Name
        loc1Name, s = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Video', '', 'AVI Videos (*.avi)')
        lineEdit1.insert(loc1Name) # Defined in Home

    def load_location2(self):
        global loc2Name
        loc2Name, s = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Locations', '', 'CSV Files (*.csv)')
        lineEdit2.insert(loc2Name) # Defined in Home

    def state_save_video(self, state):
        if state == QtCore.Qt.Checked:
            self.saveVideo = 1
            print("saveVideo =", self.saveVideo)
        else:
            self.saveVideo = 0
            print("saveVideo =", self.saveVideo)
        btn1.setEnabled(self.saveVideo)
        lineEdit1.setEnabled(self.saveVideo)

    def state_save_data(self, state):
        if state == QtCore.Qt.Checked:
            self.saveData = 1
            print("saveData =", self.saveData)
        else:
            self.saveData = 0
            print("saveData =", self.saveData)
        btn2.setEnabled(self.saveData)
        lineEdit2.setEnabled(self.saveData)

    def bbox_grab(self, frame1):
        # Receive first location of child
        bbox = cv.selectROI(frame1, False) # Selcting an ROI for child's first position
        bbox = np.array([bbox[1], bbox[0], bbox[3], bbox[2]])
        bbox_l, bbox_w = bbox[2], bbox[3] # because length and width of bbox is constant, we will define them globally here
        
        cv.destroyAllWindows()

        return bbox

    def Autism_Screening(self):
        if loc0Name == "":
            popUpM = QtWidgets.QMessageBox.information(self, 'Error', "Please load a video first")
            return None
        if self.saveVideo == 1 and loc1Name == "":
            popUpM = QtWidgets.QMessageBox.information(self, 'Error', "Please enter an output video path or uncheck \"Save Output Video\" checkbox")
            return None
        if self.saveData == 1 and loc2Name == "":
            popUpM = QtWidgets.QMessageBox.information(self, 'Error', "Please enter an output data path or uncheck \"Save Output Locations\" checkbox")
            return None

        # Load selected vidoe to a variable
        inVid = cv.VideoCapture(loc0Name)

        # Save output video in the selected path
        if self.saveVideo == 1:
            fourcc = cv.VideoWriter_fourcc(*'XVID') # Video codec parameter
            outVid = cv.VideoWriter(loc1Name, fourcc, 12.46, (1920, 1080))

        # Save output locations in the selected path
        if self.saveData == 1:
            dataFile = open(loc2Name, 'w', newline = '')
            outLoc = csv.writer(dataFile)
        
        ret, frame1= inVid.read()
        bbox = self.bbox_grab(frame1)
        position = np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]) # position of child that is position of center of the bbox
        frame1_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        
        # First frame operations
        ori_res, prv_ori, counter = 10, 0, 0
        direction = "right"
        W = np.random.randn(bbox[2]*bbox[3]*3 + 1, 2) * 0.0001
        font = cv.FONT_HERSHEY_SIMPLEX
        sysInit(frame1, bbox)
        # Playing the video
        while(inVid.isOpened()):
        #for x in range(200):
    
            ret, frame2 = inVid.read() # Reading next frame
            if ret == 0: # Check if next frame still exist or not (the video has been ended or not)
                break
            frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY) # Convert RGB image(frame) to grayscale
            
            # Check if target is going to right or left and save the state in variable named direction
            if prv_ori > 90 and prv_ori <= 270:
                direcion = "left"
            else:
                direcion = "right"
            
            inOrOut = inViewCheck(bbox) # This function is for checking if the child is in view or not
            
            if inOrOut == "Inside":
                
                ret, W = onlineSVM(frame1, bbox, W, direction) # This function trains child patch when child is in vecinity of center of view
                
                u_crop, v_crop = childFlow(frame1_gray, frame2_gray, bbox, direction)
                domOri, magHist = DomOriMag(u_crop, v_crop, ori_res)
                magHist[0:np.uint8(magHist.shape[0]/10)] = 0
                domMag = np.argmax(magHist)
                # In this part, we want to save some of our previous direction to compare with our new one and if the onlie direction
                # completely differs from what we get in our previous loops, we will ignore that and consider our saved previous
                # orientation except that
                if domMag > 50: # You can remove this if you don't want to check magnitude condition if you want to consider orientation
                                # memory for all kind of movements
                    domOri, counter = oriJumpReduct(domOri, prv_ori, counter) # To detect orientation false calculations
                else:
                    counter = 0
                
                rowMove, colMove = MagOriToMove(domMag, domOri) # This function convert (Magnitude, Orintation) data to diaplacement of 
                                                                # our bounding box
                
                outImg = np.copy(frame1)
                cv.rectangle(outImg, (bbox[1], bbox[0]), (bbox[1] + bbox[3], bbox[0] + bbox[2]), (255, 15, 255), 10)
                if self.saveVideo == 1:
                    outVid.write(outImg)
                cv.namedWindow('Out', cv.WINDOW_NORMAL)
                cv.imshow('Out', outImg)
        
                bbox += np.array([rowMove, colMove, 0, 0])
                
                ################################################
                rowComp, colComp = MoveComp(frame1_gray, frame2_gray, bbox, domMag)
        
                bbox += np.array([rowComp, colComp, 0, 0])
                ################################################
                
            elif inOrOut == "down-Out":
        
                bbox = bottomSearch(frame2, frame1_gray, frame2_gray, bbox, W)

                if self.saveVideo == 1:
                    outVid.write(frame1)
                cv.namedWindow('Out', cv.WINDOW_NORMAL)
                cv.imshow('Out', frame1)
        
            elif inOrOut == "left-Out":
        
                bbox = leftSearch(frame2, frame1_gray, frame2_gray, bbox, W)

                if self.saveVideo == 1:
                    outVid.write(frame1)
                cv.namedWindow('Out', cv.WINDOW_NORMAL)
                cv.imshow('Out', frame1)
            
            frame1_gray = frame2_gray
            frame1 = frame2
            prv_ori = domOri
            if self.saveData == 1:
                outLoc.writerow([bbox[0] + np.floor(bbox[2]/2), bbox[1] + np.floor(bbox[3]/2)])
    
            # Exit if ESC pressed
            k = cv.waitKey(1) & 0xff
            if k == 27 : break

        inVid.release()
        cv.destroyAllWindows()
        if self.saveVideo == 1:
            outVid.release()
        if self.saveData == 1:
            dataFile.close()
    
    def close_application(self):
        choice = QtWidgets.QMessageBox.question(self, 'Exit',
                                                "Are you sure you want to exit?",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            sys.exit()
        else:
            pass




app = QtWidgets.QApplication(sys.argv)
Gui = Window()
sys.exit(app.exec_())
