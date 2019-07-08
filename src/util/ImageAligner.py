import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage as sk
from scipy import ndimage
import face_recognition
import math
from PIL import Image

def CalEyesPos(image):
    face_landmarks_list = face_recognition.face_landmarks(image)
    left_eye_position = face_landmarks_list[0]['left_eye']
    right_eye_position = face_landmarks_list[0]['right_eye']
    sx = [(left_eye_position[0][0]+left_eye_position[3][0])/2,(left_eye_position[0][1]+left_eye_position[3][1])/2]
    dx = [(right_eye_position[0][0]+right_eye_position[3][0])/2,(right_eye_position[0][1]+right_eye_position[3][1])/2]
    return np.array(sx), np.array(dx)

def RotateImage(img):
    sx_pos, dx_pos = CalEyesPos(img)
    if(sx_pos[1] > dx_pos[1]):
        frecciadx = [sx_pos[0],dx_pos[1]]
        u = sx_pos-dx_pos
        v = frecciadx-dx_pos
        angle = math.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))
        rotated = ndimage.rotate(img, -angle, mode='nearest')
        return rotated
    elif(sx_pos[1] < dx_pos[1]):
        frecciasx = [dx_pos[0],sx_pos[1]]
        u = frecciasx-sx_pos
        v = dx_pos-sx_pos
        angle = math.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))
        rotated = ndimage.rotate(img, angle, mode='nearest')
        return rotated
    else:
        return img

def ScaleImage(rotated):
    scaling_factor = 36
    sx_pos, dx_pos = CalEyesPos(rotated)
    dist = dx_pos[0]-sx_pos[0]
    if(dist != scaling_factor):
        scale = scaling_factor/dist
        width = int(rotated.shape[1] * scale)
        height = int(rotated.shape[0] * scale)
        resized = cv2.resize(rotated, (width,height), interpolation = cv2.INTER_AREA)
        return resized
    else:
        return rotated

def CropImage(resized):
    sx_pos, dx_pos = CalEyesPos(resized)
    left = sx_pos[0]-70
    upper= sx_pos[1]-110
    right= left+178
    bottom = upper+218
    im = Image.fromarray(resized)
    cropped = np.array(im.crop((left, upper, right, bottom)))
    return cropped





#=================MAIN================#

path = 'marco.jpg'
img = cv2.imread(path,1)
cv2.imshow('Original',img)
result = CropImage(ScaleImage(RotateImage(img)))
cv2.imshow('Result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
