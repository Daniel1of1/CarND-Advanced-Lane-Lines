import numpy as np

# as Taken from Udacity course material
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def curvature(left_fit,right_fit):

    ploty = np.linspace(0, 719,720)

    y_eval = np.max(ploty)

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    average = (left_curverad + right_curverad) / 2

    return average

def center_offset(left_fit,right_fit,img_size):
    ploty = np.linspace(0, img_size[0]-1, img_size[0])
    y_eval = np.max(ploty)

    overlay = np.zeros((img_size)).astype(np.uint8)
    overlay = np.dstack((overlay, overlay, overlay))
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    from_center = (left_fitx[-1]+right_fitx[-1] -1280)*xm_per_pix/2
    
    return from_center
    