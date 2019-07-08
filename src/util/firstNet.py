from keras import models
from keras import layers
from keras.models import load_model
import numpy as np
import time
import Util as u
import subprocess
import sys
# controlla che i parametri da riga siano:
# Nome File / Nfeatures / startTrain / end Train / start vat /end val / epocs/ crop /model to load
assert len(sys.argv) >=9 and len(sys.argv) <= 10

fileName= sys.argv[1]
#fa comparire il titolo figo
# print(subprocess.check_output(['figlet','Face Feartures Recognition']).decode('ascii'))

img_path="../CelebA/Img/img_align_celeba"
label_path="../CelebA/Anno/list_attr_celeba.txt"
graph_path="../face-features-recognition"

print ('Number of arguments:', len(sys.argv), 'arguments.')

y,x,h,w=0,0,218,178

if(sys.argv[8]=="eyes"):
    y,x,h,w=80,0,60,178
if(sys.argv[8]=="mouth"):
    y,x,h,w=130,0,50,178
if(sys.argv[8]=="beard"):
    y,x,h,w=130,0,88,178
if(sys.argv[8]=="hat"):
    y,x,h,w=0,0,105,178


train_label=u.getLabels(label_path, [int(sys.argv[2])], True, startLine=int(sys.argv[3]), nLines=int(sys.argv[4]))
train_set =u.getImages(img_path,startImg = int(sys.argv[3]),nImg=int(sys.argv[4]), y=y,x=x,h=h,w=w)
val_label=u.getLabels(label_path, [int(sys.argv[2])], True, startLine=int(sys.argv[5]),nLines=int(sys.argv[6]))
val_set =u.getImages(img_path, startImg=int(sys.argv[5]), nImg=int(sys.argv[6]), y=y,x=x,h=h,w=w)
print("\n\nPercentuale di 1 nel train: "+ str(u.CountFeatures(train_label)))
print("Percentuale di 1 nel val: "+ str(u.CountFeatures(val_label)))

#shuffle dei dati(non ci serve)
# indices = np.arange(train_set.shape[0])
# np.random.shuffle(indices)
# train_set= train_set[indices]
# train_label = train_label[indices]
s = time.time()
print("Start time: "+str(s)+"\n\n\n===========================================================\n\n")


if(len(sys.argv) ==10):
    model = load_model(sys.argv[9])
else:
    #struttura della rete
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(h,w, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history=model.fit(train_set, train_label, validation_data=(val_set, val_label),epochs=int(sys.argv[7]), batch_size=64) #aggiusta batch size

e=time.time()
print("End time: "+str(e))
print ("Totaltime: "+str(e-s))

#salvataggio dei risultati e della rete addestrata
model.save(fileName+".h5")
u.printResult(history,True,graph_path,fileName)




# >>> test_loss, test_acc = model.evaluate(test_images, test_labels)
# >>> test_acc
# 0.99080000000000001
