import cv2
import numpy as np
import glob
import pickle


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
# Make a list of calibration images

# 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환
images = glob.glob('kakaotalk/*.jpg')
# Step through the list and search for chessboard corners

total_images = len(images)
for idx, fname in enumerate(images):
   img = cv2.imread(fname)
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # Find the chessboard corners
   ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
   # If found, add object points, image points
   if ret == True:
       objpoints.append(objp)
       imgpoints.append(corners)
       # Draw and display the corners
       cv2.drawChessboardCorners(img, (9,6), corners, ret)
       write_name = 'cal/corners_found'+str(idx)+'.jpg'
       cv2.imwrite(write_name, img)
       out_str = f'{idx}/{total_images}'
       cv2.putText(img, out_str, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
       cv2.imshow('img', img)
       cv2.waitKey(500)
cv2.destroyAllWindows()

img = cv2.imread('kakaotalk/KakaoTalk_20210929_140328435_10.jpg')
height, width = img.shape[:2]
img_size = (width, height)
# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                               img_size, None,None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('kakaotalk/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "kakaotalk/final.p", "wb" ) )
print('mtx', mtx)
print('dist', dist)

img_result = cv2.hconcat([img,dst])
img_result = cv2.pyrDown(img_result)
cv2.imshow('dst',img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()