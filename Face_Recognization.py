from PyQt5 import QtWidgets,uic
import cv2,os
import face_recognition
from PyQt5.QtGui import  QPixmap,QImage
import threading
import time,os
import numpy as np
from AddDatabase import AddDB


class FRC(QtWidgets.QMainWindow):
    def __init__(self):
        super(FRC,self).__init__()
        uic.loadUi("./Layout/Face_Recognization.ui",self)
        self.show()
        self.setFixedSize(self.size())
        self.isload=False
        self.loadtxt=[
             "Loading",
             "Loading.",
             "Loading..",
             "Loading...",
             "Loading...."
        ]
        self.loadno=0

        threading.Thread(target=self.Cam).start()
        self.getdata=threading.Thread(target=self.GetData,args=["./Database"])
        self.getdata.start()
        self.btn_2.clicked.connect(lambda:threading.Thread(target=self.GetData,args=["./Database"]).start())
        self.btn.clicked.connect(lambda:self.OpenAddNew())   


    def ISFace(self,img):
            face = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")
            gre = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            face123 = face.detectMultiScale(gre,1.1,4)
            # print(face123)
            return True if len(face123)>0 else False

    def OpenAddNew(self):
        window1=AddDB(self).show()
        # window1.exec_()
  

    def GetData(self,Images_Folder_Path):
        self. ALL_FILE=[]
        self.PERSON_NAME=[] 
        self.isload=True
        images = os.listdir(Images_Folder_Path)
        for img in images:
            curimg = cv2.imread(f'{Images_Folder_Path}/{img}')
            self.ALL_FILE.append(curimg)
            self.PERSON_NAME.append(os.path.splitext(img)[0])
        self.find_endcoding(self.ALL_FILE)
        self.isload=False
        
    def find_endcoding(self,images):
        self.Known=[]
        for img in images:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(img)[0]
            self.Known.append(encoding)
        
    def atten_face(self,Image,img):
        self.name=None
        image5 = cv2.resize(Image,(0,0),None,0.25,0.25)
        image5 = cv2.cvtColor(image5,cv2.COLOR_BGR2RGB)

        loc = face_recognition.face_locations(image5)
        end = face_recognition.face_encodings(image5,loc)

        for ef,locv in zip(end,loc):
            match = face_recognition.compare_faces(self.Known,ef)
            facedistanse = face_recognition.face_distance(self.Known,ef)
            mi = np.argmin(facedistanse)

            if match[mi]:
                self.name = self.PERSON_NAME[mi].upper()
                y1,x2,y2,x1 = locv
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255))
                cv2.putText(img,str(self.name),(x1,y2+30),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,255))
            else:
                self.name="Unknow"
                y1,x2,y2,x1 = locv
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255))
                cv2.putText(img,"Unknow",(x1,y2+30),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,255))
            
        return img,self.name
                

    def Cam(self):
        # self.getdata.join()
        vid=cv2.VideoCapture(0)
        while vid:
            if self.isload:
                self.Image.hide()
                self.load.show()
                if self.loadno==0:
                    self.load.setText(self.loadtxt[1])
                    self.loadno=1
                    time.sleep(0.5)
                elif self.loadno==1:
                    self.load.setText(self.loadtxt[2])
                    self.loadno=2
                    time.sleep(0.5)
                elif self.loadno==2:
                    self.load.setText(self.loadtxt[3])
                    self.loadno=3
                    time.sleep(0.5)
                elif self.loadno==3:
                    self.load.setText(self.loadtxt[4])
                    self.loadno=4
                    time.sleep(0.5)
                elif self.loadno==4:
                    self.load.setText(self.loadtxt[0])
                    self.loadno=0
                    time.sleep(0.5)
            else:
                self.load.hide()
                self.Image.show()
                _,img=vid.read()
                if _:
                    self.name=None
                    img=cv2.resize(img,(521, 441))
                    height, width, channel = img.shape
                    if self.ISFace(img) and not self.isload:
                        img,self.name=self.atten_face(img,img)
                    if self.name:
                        self.Name.setText(self.name)
                    else:
                        self.Name.setText("")

                    qImg = QImage(img.data, width, height, width*3, QImage.Format_RGB888).rgbSwapped()
                    pixmap=QPixmap(qImg)
                    self.Image.setPixmap(pixmap)


    



app=QtWidgets.QApplication([])
window=FRC()
app.exec_()