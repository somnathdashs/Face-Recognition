from PyQt5 import QtWidgets,uic,QtCore
import cv2
import face_recognition_models
from PyQt5.QtGui import QIcon, QPixmap,QImage
import sys
import threading
import time
class AddDB(QtWidgets.QMainWindow):
    def __init__(self,parent=None):
        super(AddDB,self).__init__(parent)
        uic.loadUi("./Layout/AddDastabase.ui",self)
        self.setFixedSize(self.size())
        self.setup()

    def setup(self):
        self.NameField=self.nametxt
        self.Savebtn=self.pushButton
        threading.Thread(target=self.load_cam).start()
        self.Savebtn.clicked.connect(lambda:self.Save(self.frame))
        
    def Save(self,imageadd):
        if self.ISFace(imageadd):
            cv2.imwrite(f"./Database/{self.NameField.text()}.jpg",imageadd)
            self.info.setText("Image Saved")
            self.NameField.setText("")
            
        else:
            self.info.setText("Face is not detected. Can't save")

    
    def load_cam(self):
        vid=cv2.VideoCapture(0)
        while vid:
            _,self.frame=vid.read()
            imageadd=cv2.resize(self.frame,(481, 511))
            imageadd=self.detec_face(imageadd)
            height, width, channel = imageadd.shape
            bytesPerLine = 3 * width
            qimageadd = QImage(imageadd.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap=QPixmap(qimageadd)
            self.Image.setPixmap(pixmap)


    def ISFace(self,imageadd):
        face = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")
        gre = cv2.cvtColor(imageadd,cv2.COLOR_BGR2GRAY)
        face123 = face.detectMultiScale(gre,1.1,4)
        # print(face123)
        return True if len(face123)>0 else False


    
    def detec_face(self,imageadd):
        face = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")
        gre = cv2.cvtColor(imageadd,cv2.COLOR_BGR2GRAY)
        face123 = face.detectMultiScale(gre,1.1,4)
        # vid = cv2.VideoCapture(0)
        for(x,y,w,h) in face123 :
            cv2.rectangle(imageadd,(x,y),(x+w,y+h),(225,0,0),2)

        return imageadd 








if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    window=AddDB().show()
    sys.exit(app.exec())