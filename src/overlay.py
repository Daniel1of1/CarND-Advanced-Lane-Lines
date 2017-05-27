import numpy as np
import cv2
import src.lane_measurements


ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def lane_measurements_overlay(left_fit,right_fit,img_size):
    overlay = np.zeros((img_size)).astype(np.uint8)
    overlay = np.dstack((overlay, overlay, overlay))
    
    curvature = lane_measurements.curvature(left_fit,right_fit)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(overlay, "lane curvature radius:", (30, 60), font, 1, (255,255,255), 2,cv2.LINE_AA)
    cv2.putText(overlay, "{0:.2f}m".format(curvature), (30, 100), font, 1, (255,255,255), 2,cv2.LINE_AA)


    from_center = lane_measurements.center_offset(left_fit,right_fit,img_size)
    
    cv2.putText(overlay, "center offset:", (30, 200), font, 1, (255,255,255), 2,cv2.LINE_AA)
    center_offset_string = "{0:.2f}m left of center".format(from_center)
    cv2.putText(overlay, center_offset_string ,(30,240), font, 1, (255,255,255),2,cv2.LINE_AA)

    return overlay

def lane_overlay(left_fit,right_fit,img_size,c1=[100,0,0],c2=[0,100,0],c3=[0,0,100]):
    ploty = np.linspace(0, img_size[0]-1, img_size[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    overlay = np.zeros((img_size)).astype(np.uint8)
    overlay = np.dstack((overlay, overlay, overlay))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    r_line_pts = pts_right.reshape((-1,1,2))
    l_line_pts = pts_left.reshape((-1,1,2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(overlay, np.int_([pts]), (0,100, 0))
    cv2.polylines(overlay,np.int_([r_line_pts]),False,(255,0,0), 25)
    cv2.polylines(overlay,np.int_([l_line_pts]),False,(255,0,0), 25)
    return overlay
