from keras import models
from keras import layers
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import Adam
import numpy as np
import time
import Util as u
import subprocess
import sys
from DataSequence import DataSequence
# controlla che i parametri da riga siano: Nome File / model to load
assert len(sys.argv) >= 2 and len(sys.argv) <= 3

fileName= sys.argv[1]
img_path="../CelebA/Img/img_align_celeba"
label_path="../CelebA/Anno/list_attr_celeba.txt"
graph_path="../face-features-recognition"

#MENU
# print(subprocess.check_output(['figlet','F F R ']).decode('ascii'))
print("Inserisci i parametri per l'addestramento:\n")
epocs, crop = input("[Epocs] [Crop]").split()
features = input("[features]")
features=[int(s) for s in features.split()]
print("\n")
print("Inserisci i parametri per i dati di train:\n")
startTrain, endTrain, startRow, numRow = input("[startTrain(min 1)] [endTrain] [startRow(min 0)] [numRow]").split()
print("\n")
print("Inserisci i parametri per i dati di validation:\n")
startVal, endVal, startRowVal, numRowVal = input("[startVal] [endVal] [startRowVal(min 0)] [numRowVal]").split()
filterFeatures= input("[filterFeatures (0 for no feature)(use ',' to split )]")

if( ( not "," in filterFeatures) and int(filterFeatures)==0):
    # print("if")
    (train_label, listImg)=u.getOneHotLabel(label_path, features , startLine=int(startTrain), endLine=int(endTrain), startRow=int(startRow), numRow=int(numRow))
    (val_label, valImg)=u.getOneHotLabel(label_path, features, startLine=int(startVal),endLine=int(endVal), startRow=int(startRowVal), numRow=int(numRowVal))
else:
    # print("else")
    valueFeatures= input("[valueFeatures]")
    filterFeatures= [int(s) for s in filterFeatures.split(',')]
    valueFeatures= [int(s) for s in valueFeatures.split(',')]
    #TODO
    (train_label, listImg)=u.getFilteredOneHotLabel(label_path, filterFeatures, valueFeatures, features, startLine=int(startTrain), endLine=int(endTrain), startRow=int(startRow), numRow=int(numRow))
    (val_label, valImg)=u.getFilteredOneHotLabel(label_path, filterFeatures, valueFeatures, features, startLine=int(startVal),endLine=int(endVal), startRow=int(startRowVal), numRow=int(numRowVal))




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

else:
    print("Invalid Crop value")
    sys.exit(1)


# (train_label, listImg)=u.getOneHotLabel(label_path, features , startLine=int(startTrain), endLine=int(endTrain), startRow=int(startRow), numRow=int(numRow))
print("\n\nPercentuale di 1 nel train: "+ str(u.CountFeatures(train_label)))
train_label=to_categorical(train_label)
print(train_label[0])
# train_set =u.getImagesFromList(img_path, listImg, y=y,x=x,h=h,w=w)
# (val_label, valImg)=u.getOneHotLabel(label_path, features, startLine=int(startVal),endLine=int(endVal), startRow=int(startRowVal), numRow=int(numRowVal))
print("Percentuale di 1 nel val: "+ str(u.CountFeatures(val_label)))
val_label=to_categorical(val_label)
# val_set =u.getImagesFromList(img_path, valImg, y=y,x=x,h=h,w=w)

batch_size=64
train_sequence= DataSequence(listImg,train_label,batch_size,img_path,y,x,h,w,1)
val_sequence= DataSequence(valImg,val_label,batch_size,img_path,y,x,h,w,1)

#shuffle dei dati(non ci serve)
# indices = np.arange(train_set.shape[0])
# np.random.shuffle(indices)
# train_set= train_set[indices]
# train_label = train_label[indices]
s = time.time()
print("Start time: "+str(s)+"\n\n\n===========================================================\n\n")


if(len(sys.argv) ==3):
    model = load_model(sys.argv[2])
else:
    #struttura della rete
    model = models.Sequential()
    model.add(layers.Conv2D(32, (7, 7), activation='relu', input_shape=(h,w, 3), padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(len(features), activation='softmax'))
    opt=Adam(lr=0.001,decay=1e-8)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['acc'])
    # model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

# history=model.fit(train_set, train_label, validation_data=(val_set, val_label),epochs=int(epocs), batch_size=64) #aggiusta batch size
history=model.fit_generator(train_sequence,epochs=int(epocs), validation_data=val_sequence) #aggiusta batch size

print(model.summary())  #TODO
e=time.time()
print("End time: "+str(e))
print ("Totaltime: "+str(e-s))

#salvataggio dei risultati e della rete addestrata
model.save(fileName+".h5")
u.printResult(history,True,graph_path,fileName)




# >>> test_loss, test_acc = model.evaluate(test_images, test_labels)
# >>> test_acc
# 0.99080000000000001
