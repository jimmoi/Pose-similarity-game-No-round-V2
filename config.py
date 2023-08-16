import os
image_scale = 1500 #1000 mean lenght of height image
round_pose = 5 #----> less than number of image
scale = 1 #default is 1 ---->round 2:4k | 4/3:2k | 1:FHD **is depends on your webcam res
cam_type = "Hor" #"Hor = Horizontal cam, Ver = Vertical cam"
cam_number = 0 #default is 0 choose your camera devices
crop_Hor = 480
crop_Ver = 0
text_size,text_thick,x,y = 3,3,10,60
window_name = "Game pose"


























Parent_folder = os.path.dirname(__file__) + "\\"
image_folder = Parent_folder + "raw ref img\\put image here"
output_all_image = Parent_folder + "reference image deletable"
output_all_point = Parent_folder + "all_reference_poses_point_deletable.py"


# Parent_folder = "A:\mediapipe\Pose-similarity-game-freaking-last-ver"
# Parent_folder = Parent_folder + "\\"
# image_folder_name = "raw ref img\\put image here"#don't change
# image_folder = Parent_folder+image_folder_name
# output_folder_name = "reference image deletable"#don't change
# output_all_image = Parent_folder+output_folder_name
# reference_file_name = "all_reference_poses_point_deletable.py" #don't change
# output_all_point = Parent_folder+reference_file_name
