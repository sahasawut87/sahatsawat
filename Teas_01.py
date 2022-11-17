import numpy as np
import cv2
import os
import cv2.aruco as aruco 
img = cv2.imread("sahatsawat\Template-2.jpg")
vdo = cv2.VideoCapture("sahatsawat\left_output-1.mp4")
img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

min_match_number = 15
sift = cv2.SIFT_create()
bf = cv2.BFMatcher() 

def findAruco(img,marker_suze=6,total_marker=250,drae=True):
 
    key = getattr(aruco,f'DICT_(marker_size)X(marker_size)_(total_marker)')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bbox,ids,_=aruco.detecMarkers(img_gray,arucoDict,paramrters=arucoParam)
    print(ids)


while vdo.isOpened() :
    ret, frame = vdo.read()
    
    if ret :
        frame_array =np.array([[0,0,0],[0,1,0],[0,0,0]])
        frame_filter = cv2.filter2D(frame, -1, frame_array)
        frame_gray = cv2.cvtColor(frame_filter, cv2.COLOR_BGR2GRAY)

        template_kpts, template_desc = sift.detectAndCompute(img_gray, None)
        query_kpts, query_desc = sift.detectAndCompute(frame_gray, None)
        matches = bf.knnMatch(template_desc, query_desc, k=2)
        good_matches = list()
        good_matches1 = list()
        for m, n in matches :
            if m.distance < 0.7*n.distance :
                good_matches.append(m)
                # print(good_matches)
                good_matches1.append([m])
        if len(good_matches) > min_match_number :
            src_pts = np.float32([ template_kpts[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ query_kpts[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            src_pts,dst_pts = np.float32((src_pts,dst_pts)) 

            H, inlier_masks = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.2) # H RANSAC
            # get the bounding box around template image
            h, w = img_gray.shape[:2]
            template_box = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2)
            transformed_box = cv2.perspectiveTransform(template_box, H)
            
            detected_img = cv2.polylines(frame, [np.int32(transformed_box)], True, (0,0,255), 3, cv2.LINE_AA)
            detected_img2 = cv2.drawMatchesKnn(img, template_kpts, detected_img, query_kpts, good_matches1, None, flags=2, matchesMask= inlier_masks)
            # cv2.imshow('Video frame', detected_img2)
            # findAruco(img)
            cv2.imshow('Video frame', frame)
        if cv2.waitKey(53) & 0xFF == ord('q'): # this line control the period between image frame
            break
    # else :
    #     break
# vdo.release()
# cv2.destroyAllWindows()


