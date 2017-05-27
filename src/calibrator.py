import numpy as np
import cv2

class Calibrator:
    
    objpoints = []
    imgpoints = []

    def __init__(self,imgs, nx, ny):
        objp = np.zeros((ny*nx,3),np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)


        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret == True:
                self.imgpoints.append(corners)
                self.objpoints.append(objp)
                
                
    def undistort(self,img):
    
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
    
        return dst


