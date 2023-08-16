from config import round_pose, scale, cam_type, crop_Hor, crop_Ver, output_all_image, cam_number
from config import text_size,text_thick,x,y
import os
import cv2
import mediapipe as mp
import numpy as np
from all_reference_poses_point_deletable import reference_pose
import needed_func as nf
import time

#Initialize pose mediapipe and OpenCV
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#set webcam max resolution
cap = cv2.VideoCapture(cam_number)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,2000);  cap.set(cv2.CAP_PROP_FRAME_WIDTH,2000)

w_cam = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h_cam = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
win_cam = (int(w_cam),int(h_cam))
win_monitor = (int(w_cam*scale),int(h_cam*scale))

#get drawed image in folder
filename = os.listdir(output_all_image)


#random image 
# image_passed = set()
# if len(filename) >= round_pose:
#     while True:
#         random_image = np.random.randint(0, len(filename))
#         image_passed.add(random_image)
#         if len(image_passed) >= round_pose:  
#             break
# else:
#     print("round_pose must less than number of image")




# filter image is png or jpg, and cv2 read image and store in image_readed
# image_readed store ==> filename, reference_pose_xy | image_readed, width, height 
image_chosen_readed,image_chosen_point = [],[]
for i in range(len(filename)):
    if filename[i].endswith(".jpg") or filename[i].endswith(".png"):   
        image_path = os.path.join(output_all_image, filename[i])          
        image = cv2.imread(image_path)
        image_chosen_readed.append(image)
        image_chosen_point.append(reference_pose[i])


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    image_campiang = np.random.randint(0, len(filename))
    reward_time = None
    frame_point = [(0,0)*33]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to open webcam.")
            exit()

        #frame to show in window
        frame = nf.cam_dir(cam_type,win_cam,win_monitor,crop_Ver,crop_Hor,frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #process and draw on image
        results = pose.process(rgb_frame)
        if results.pose_landmarks: #draw pose landmark on your webcam
            pose_landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]

            #get ref and frame point
            ref_point = image_chosen_point[image_campiang]
            frame_prop = frame.shape[0:2]
            frame_point = pose_landmarks

            #use preason colliretion
            similarity = nf.preacole_shape(ref_point,frame_point)
    
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                                )
            y0, dy = 50, 4
            if similarity[0] > 0.97:
                color_code = (0,139,0)
                cv2.putText(frame, f"Similarity: {(similarity[0]*10):.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,text_size,(color_code),text_thick,cv2.LINE_AA)
                cv2.putText(frame, "   Good Job, Here your reward", (x, y+100), cv2.FONT_HERSHEY_SIMPLEX,text_size,(color_code),text_thick,cv2.LINE_AA)
            elif similarity[0] > 0.92:
                color_code = (0,211,197)
                cv2.putText(frame, f"Similarity: {(similarity[0]*10):.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,text_size,(color_code),text_thick,cv2.LINE_AA)
                cv2.putText(frame, "   almost done", (x, y+100), cv2.FONT_HERSHEY_SIMPLEX,text_size,(color_code),text_thick,cv2.LINE_AA)
            elif similarity[0] > 0.85:
                color_code = (0,235,254)
                cv2.putText(frame, f"Similarity: {(similarity[0]*10):.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,text_size,(color_code),text_thick,cv2.LINE_AA)
                cv2.putText(frame, "   You have chance", (x, y+100), cv2.FONT_HERSHEY_SIMPLEX,text_size,(color_code),text_thick,cv2.LINE_AA)
            elif similarity[0] > 0.80:
                color_code = (0,158,254)
                cv2.putText(frame, f"Similarity: {(similarity[0]*10):.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,text_size,(color_code),text_thick,cv2.LINE_AA)
                cv2.putText(frame, "   Let's start", (x, y+100), cv2.FONT_HERSHEY_SIMPLEX,text_size,(color_code),text_thick,cv2.LINE_AA)
            else:
                color_code = (0,15,254)
                cv2.putText(frame, f"Similarity: {(similarity[0]*10):.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,text_size,(color_code),text_thick,cv2.LINE_AA)
                cv2.putText(frame, "   Hey you over there, You wanna try this game", (x, y+100), cv2.FONT_HERSHEY_SIMPLEX,text_size,(color_code),text_thick,cv2.LINE_AA)



        #insert ref image   
        ref = image_chosen_readed[image_campiang]
        ref = nf.resize_image_ref(win_monitor,ref)
        
        #show window 
        im_h = cv2.hconcat([ref,frame]) #link 2 image
        cv2.imshow("Pose Estimation", im_h)

        if cv2.waitKey(1) & 0xFF == ord("n"):
            image_campiang = np.random.randint(0, len(filename))
            continue

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break



    cap.release()
    cv2.destroyAllWindows()
        

