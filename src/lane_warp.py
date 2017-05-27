import numpy as np
import cv2

class Transformer:
    
    trap = np.array([[250,670],[(1280/2)-50,450],[(1280/2)+50,450],[1280-250,670]],dtype=np.int32)
    dst = np.array([[300,720],[300,0],[1280-300,0],[1280-300,720]],dtype=np.int32)
                
                
    def transform(self,img):

        src = np.float32(self.trap)

        dst = np.float32(self.dst)

        M = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)

        return warped

    def inverse_transform(self,img):

        src = np.float32(self.trap)

        dst = np.float32(self.dst)

        M = cv2.getPerspectiveTransform(dst, src)

        warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)

        return warped
