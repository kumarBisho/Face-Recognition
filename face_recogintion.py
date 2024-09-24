from deepface import DeepFace
import numpy as np
img_path1='/home/hitech/Documents/amit_pankaj2.jpg'

dfs=DeepFace.find(img_path=img_path1,db_path='face_recognition_db')
print("original Image:")
print(img_path1)
print("detected image from database:")
print(dfs)
# dfs=np.array(dfs)

# print("Closest image of original image:")
# print(dfs[0][0])
