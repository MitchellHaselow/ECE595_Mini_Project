# Use DeepFace module to identify faces: https://pypi.org/project/deepface/
# Used on both the training and testing sets.
from deepface import DeepFace 
import cv2
import numpy as np
import os, sys
from multiprocessing import Pool

input_path = 'test'
output_path = 'test_processed'

def img_id_to_processed_img(img_id):
    try:
        detected_aligned_face = DeepFace.extract_faces(img_path = input_path + '/' + img_id, detector_backend = 'ccv', target_size=(160, 160))
        if len(detected_aligned_face) == 1:
            resized_img  =255*cv2.resize(detected_aligned_face[0]['face'], (64,64))
            cv2.imwrite(output_path + '/' + img_id, cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    except:
        pass

if __name__ == "__main__":
    remainder = list(set(os.listdir(input_path))-set(os.listdir(output_path)))
    with Pool() as p:
        p.map(img_id_to_processed_img, remainder)
