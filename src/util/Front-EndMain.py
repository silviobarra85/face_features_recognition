import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage as sk
from scipy import ndimage
import face_recognition
import math
from PIL import Image
from keras.models import load_model
from keras import models
import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)
#tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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



def ReadImg(listPath):

    images = np.array([CropImage(ScaleImage(RotateImage(cv2.imread(path,1)))) for path in listPath])
    grayimages=np.array([cv2.cvtColor(col, cv2.COLOR_BGR2GRAY) for col in images])

    for i in range(len(images)):
        cv2.imshow(""+listPath[i] , images[i])

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    images =images.astype('float32') / 255
    grayimages= grayimages.astype('float32') / 255
    grayimages=grayimages.reshape(grayimages.shape[0], grayimages.shape[1], grayimages.shape[2],1)

    return images, grayimages

def CropImgForPred(listImg, crop):
    y,x,h,w=0,0,218,178
    if(crop=="eyes"):
        y,x,h,w=80,0,60,178
    elif(crop=="mouth"):
        y,x,h,w=130,0,50,178
    elif(crop=="none"):
        y,x,h,w=0,0,218,178
    elif(crop=="beard"):
        y,x,h,w=130,0,88,178
    elif(crop=="hat"):
        y,x,h,w=0,0,105,178

    return listImg[:,y:y+h, x:x+w]

# ----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------main-------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

#stampo il logo del progetto
# try:
#     os.system("toilet -f future  face feature recognition --filter border:metal")
# except:
#     print("Attenzione: forse non hai installato toilet!\nPer installare digita \" yay -S toilet\"")

#mi prendo i dati in input
print("Inserisci il path delle foto da elaborare:")
listPath=[]
path= ""
while(True):
    path = input("[path (0 = fine inserimento)]")
    if(path != "0"):
        listPath.append(path)
    else:
        break
print("\n")

if(len(listPath)==0):
    print("Nessuna immagine inserita")
    exit(0)


print("Inserisci il path del file di output")
path_file = input("[ path (0 = non salvare su file)]")
print("\n")
# -------------------------------------lettura immagini e caricamento dei modelli----------------------------------------------------

(colorImg, grayImg)=ReadImg(listPath)
modelArchedEyebrow=load_model('./AllResult/ArchedEyebrows/archedEyebrow_new_4.h5')
print('Loaded Arched Eyebrow model')
modelBald=load_model('./AllResult/Bald/bald_new_3.h5')
print('Loaded Bald model')
modelBangs=load_model('./AllResult/Bangs/bangs_new_3.h5')
print('Loaded Bangs model')
modelBrushyEyebrows=load_model('./AllResult/BrushyEyebrows/brushyeyebrows_new_2.h5')
print('Loaded Brushy Eyebrow model')
modelCapelli=load_model('./AllResult/Capelli/capelli_new_4.h5')
print('Loaded Capelli model')
modelEyeglasses=load_model('./AllResult/Eyeglasses/eyeglasses_new_2.h5')
print('Loaded Eyeglasses model')
modelGoatee=load_model('./AllResult/Goatee/gotee_new_2_beard.h5')
print('Loaded Goatee model')
modelHat=load_model('./AllResult/Hat/hat_new_4.h5')
print('Loaded Hat model')
modelLipstick=load_model('./AllResult/Lipstick/lipstick_new_2.h5')
print('Loaded Lipstick model')
modelMale=load_model('./AllResult/Male/male_new_1.h5')
print('Loaded Male model')
modelMouthSlightlyOpen=load_model('./AllResult/MouthSlightlyOpen/mouthOpen_new_2.h5')
print('Loaded Mouth Slightly Open model')
modelNeckTie=load_model('./AllResult/NeckTie/necktie_new_3.h5')
print('Loaded Necktie model')
modelNoBeard=load_model('./AllResult/NoBeard/noBeard_new_2.h5')
print('Loaded No Beard model')
modelSmiling=load_model('./AllResult/Smiling/smiling_new_2.h5')
print('Loaded Smiling model')
features= np.zeros([colorImg.shape[0], 17])


# ----------------------------------immagine intera-------------------------------------
MalePrediction=modelMale.predict(grayImg).reshape(grayImg.shape[0])
MalePrediction = (MalePrediction > 0.5).astype(np.int_)
features[:,10]=MalePrediction
# print('Male '+ str(MalePrediction))
# print()


# -----------------------------------hat-------------------------------------------------
grayImgHat=CropImgForPred(grayImg, 'hat')
HatPrediction= modelHat.predict(grayImgHat).reshape(grayImgHat.shape[0])
HatPrediction = (HatPrediction > 0.5).astype(np.int_)
features[:,14]=HatPrediction
# print('Hat '+ str(HatPrediction))

# TODO: Capelli
BaldPrediction=modelBald.predict(grayImgHat).reshape(grayImgHat.shape[0])
BaldPrediction = (BaldPrediction > 0.5).astype(np.int_)
features[:,1]=BaldPrediction
boolBald=(BaldPrediction<0.5)
boolBald=np.where(boolBald)

#TODO:~~~~~~~~~~~~CONTROLLARE SE FUNZIONA QUANDO TIENI QUALCUNO CHE È CALVO
colImgHat=CropImgForPred(colorImg, 'hat')
if(len(colImgHat[boolBald[0]])!=0):
    CapelliPrediction=modelCapelli.predict(colImgHat[boolBald[0]])
predictionAllCapelli=np.zeros([BaldPrediction.shape[0], 4])
if(len(colImgHat[boolBald[0]])!=0):
    predictionAllCapelli[boolBald]=CapelliPrediction
# print(predictionAllCapelli)
for i in range(len(predictionAllCapelli)):
    predictionAllCapelli[i]= (predictionAllCapelli[i] > 0.5).astype(np.int_)
features[:,3]=predictionAllCapelli[:,0]
features[:,4]=predictionAllCapelli[:,1]
features[:,5]=predictionAllCapelli[:,2]
features[:,9]=predictionAllCapelli[:,3]

if(len(grayImgHat[boolBald[0]])!=0):
    BangsPrediction=modelBangs.predict(grayImgHat[boolBald[0]]).reshape(grayImgHat[boolBald[0]].shape[0])
predictionBangsAll=np.zeros(BaldPrediction.shape[0])
if(len(grayImgHat[boolBald[0]])!=0):
    predictionBangsAll[boolBald]=BangsPrediction
predictionBangsAll = (predictionBangsAll > 0.5).astype(np.int_)
features[:,2]= predictionBangsAll

# print('Bald '+ str(BaldPrediction))
# print('Bangs '+ str(predictionBangsAll))
# print('Capelli '+ str(predictionAllCapelli))
# print()

#-------------------------------------eyes-------------------------------------------
grayImgEye=CropImgForPred(grayImg, 'eyes')
ArchedEyebrowsPrediction=modelArchedEyebrow.predict(grayImgEye).reshape(grayImgEye.shape[0])
ArchedEyebrowsPrediction = (ArchedEyebrowsPrediction > 0.5).astype(np.int_)
features[:,0]=ArchedEyebrowsPrediction

BrushyEyebrowsPrediction=modelBrushyEyebrows.predict(grayImgEye).reshape(grayImgEye.shape[0])
BrushyEyebrowsPrediction = (BrushyEyebrowsPrediction > 0.5).astype(np.int_)
features[:,6]=BrushyEyebrowsPrediction

EyeglassesPrediction=modelEyeglasses.predict(grayImgEye).reshape(grayImgEye.shape[0])
EyeglassesPrediction = (EyeglassesPrediction > 0.5).astype(np.int_)
features[:,7]=EyeglassesPrediction

# print('ArchedEyebrows '+ str(ArchedEyebrowsPrediction))
# print('BrushyEyebrows '+ str(BrushyEyebrowsPrediction))
# print('Eyeglasses '+ str(EyeglassesPrediction))
# print()

#---------------------------------mouth----------------------------------------------
grayImgMouth=CropImgForPred(grayImg, 'mouth')
colImgMouth=CropImgForPred(colorImg, 'mouth')
LipstickPrediction=modelLipstick.predict(colImgMouth).reshape(colorImg.shape[0])
LipstickPrediction = (LipstickPrediction > 0.5).astype(np.int_)
features[:,15]=LipstickPrediction

SmilingPrediction=modelSmiling.predict(grayImgMouth).reshape(grayImgMouth.shape[0])
SmilingPrediction = (SmilingPrediction > 0.5).astype(np.int_)
features[:,13]=SmilingPrediction

MouthSlightlyOpenPrediction=modelMouthSlightlyOpen.predict(grayImgMouth).reshape(grayImgMouth.shape[0])
MouthSlightlyOpenPrediction= (MouthSlightlyOpenPrediction > 0.5).astype(np.int_)
features[:,11]=MouthSlightlyOpenPrediction

# print('Lipstick '+ str(LipstickPrediction))
# print('Smiling '+ str(SmilingPrediction))
# print('MouthSlightlyOpen '+ str(MouthSlightlyOpenPrediction))
# print()

#----------------------------beard---------------------------------------------------
grayImgBeard=CropImgForPred(grayImg, 'beard')
boolMale=(MalePrediction.reshape(MalePrediction.shape[0])>0.5)
boolMale=np.where(boolMale)
if(len(grayImgBeard[boolMale[0]])!=0):
    NoBeardPrediction=modelNoBeard.predict(grayImgBeard[boolMale[0]]).reshape(grayImgBeard[boolMale[0]].shape[0])
predictionBeardAll=np.ones(MalePrediction.shape[0])
if(len(grayImgBeard[boolMale[0]])!=0):
    predictionBeardAll[boolMale]=NoBeardPrediction
predictionBeardAll = (predictionBeardAll > 0.5).astype(np.int_)
features[:,12]=predictionBeardAll

boolNoBeard=(predictionBeardAll.reshape(predictionBeardAll.shape[0])<0.5)
boolNoBeard=np.where(boolNoBeard)
if(len(grayImgBeard[boolNoBeard[0]])!=0):
    GoateePredicition=modelGoatee.predict(grayImgBeard[boolNoBeard[0]]).reshape(grayImgBeard[boolNoBeard[0]].shape[0])
predictionGoateeAll=np.zeros(predictionBeardAll.shape[0])
if(len(grayImgBeard[boolNoBeard[0]])!=0):
    predictionGoateeAll[boolNoBeard]=GoateePredicition
predictionGoateeAll = (predictionGoateeAll > 0.5).astype(np.int_)
features[:,8]=predictionGoateeAll

NeckTiePrediction=modelNeckTie.predict(grayImgBeard).reshape(grayImgBeard.shape[0])
NeckTiePrediction = (NeckTiePrediction > 0.5).astype(np.int_)
features[:,16]=NeckTiePrediction
# print('NoBeard '+ str(predictionBeardAll))
# print('Goatee '+ str(predictionGoateeAll))
# print('NeckTie '+ str(NeckTiePrediction))
# print()
# print(features)

# output un vettore di 0 1 in base alle Features
# features = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
features= features.astype(np.int_)
print('sono qiu')
if(path_file != "0"):
    output = open(path_file, "a")
    for i in range(0,len(listPath)):
        print("\n\n",file=output)
        print("Immagine di riferimento: "+listPath[i]+"\n\n",file=output)
        print("|=================|======|=======|============|=============|============|================|============|========|===========|======|=====================|==========|=========|=============|==================|=================|",file=output)
        print("| Arched_Eyebrows | Bald | Bangs | Black_Hair | Blonde_Hair | Brown_Hair | Bushy_Eyebrows | Eyeglasses | Goatee | Gray_Hair | Male | Mouth_Slightly_Open | No_Beard | Smiling | Wearing_Hat | Wearing_Lipstick | Wearing_Necktie |",file=output)
        print("|=================|======|=======|============|=============|============|================|============|========|===========|======|=====================|==========|=========|=============|==================|=================|",file=output)
        print("| "+str(features[i][0])+"               | "+str(features[i][1])+"    | "+str(features[i][2])+"     | "+str(features[i][3])+"          | "+str(features[i][4])+"           | "+str(features[i][5])+"          | "+str(features[i][6])+"              | "+str(features[i][7])+"          | "+str(features[i][8])+"      | "+str(features[i][9])+"         | "+str(features[i][10])+"    | "+str(features[i][11])+"                   | "+str(features[i][12])+"        | "+str(features[i][13])+"       | "+str(features[i][14])+"           | "+str(features[i][15])+"                | "+str(features[i][16])+"               |",file=output)
        print("|=================|======|=======|============|=============|============|================|============|========|===========|======|=====================|==========|=========|=============|==================|=================|",file=output)
        print("File Saved.\n")
else:
    for i in range(0,len(listPath)):
        print("\n\n")
        print("Immagine di riferimento: "+listPath[i]+"\n\n")
        print("|=====================|=======|")
        print("| Arched_Eyebrows     |"+str(features[i][0])+"      |")
        print("| Bald                |"+str(features[i][1])+"      |")
        print("| Bangs               |"+str(features[i][2])+"      |")
        print("| Black_Hair          |"+str(features[i][3])+"      |")
        print("| Blonde_Hair         |"+str(features[i][4])+"      |")
        print("| Brown_Hair          |"+str(features[i][5])+"      |")
        print("| Brushy_Eyebrows     |"+str(features[i][6])+"      |")
        print("| Eyeglasses          |"+str(features[i][7])+"      |")
        print("| Goatee              |"+str(features[i][8])+"      |")
        print("| Gray_Hair           |"+str(features[i][9])+"      |")
        print("| Male                |"+str(features[i][10])+"      |")
        print("| Mouth_Slightly_Open |"+str(features[i][11])+"      |")
        print("| No_Beard            |"+str(features[i][12])+"      |")
        print("| Smiling             |"+str(features[i][13])+"      |")
        print("| Wearing_Hat         |"+str(features[i][14])+"      |")
        print("| Wearing_Lipstick    |"+str(features[i][15])+"      |")
        print("| Wearing_Necktie     |"+str(features[i][16])+"      |")
        print("|=====================|=======|\n\n")
