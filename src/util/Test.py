from keras.models import load_model
from keras import models
import time
import sys
import Util as u

# fileName /num feature/ crop
assert len(sys.argv) ==2
fileName= sys.argv[1]
model=load_model(fileName)

img_path="../CelebA/Img/img_align_celeba"
label_path="../CelebA/Anno/list_attr_celeba.txt"

print("Inserisci i parametri per l'addestramento:\n")
feature, crop, bw = input("[Feature] [Crop] [Grayscale(0 si, 1 no)]").split()
filterFeatures= input("[filterFeatures (0 for no feature)(use ',' to split )]")


if( ( not "," in filterFeatures) and int(filterFeatures)==0):
    # print("if")
    (test_label, listImg)=u.getBalancedLabel(label_path, int(feature), True, startLine=182638, endLine=202599, startRow=0, numRow=0)
else:
    # print("else")
    valueFeatures= input("[valueFeatures]")
    filterFeatures= [int(s) for s in filterFeatures.split(',')]
    valueFeatures= [int(s) for s in valueFeatures.split(',')]
    (test_label, listImg)=u.getFilteredBalLabel(label_path, filterFeatures, valueFeatures, int(feature), True, startLine=182638, endLine=202599, startRow=0, numRow=0)


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


bw=int(bw)
# test_label=u.getLabels(label_path, [int(sys.argv[2])], True, startLine=182638,nLines=202599)
# test_set =u.getImages(img_path,startImg=182638, nImg= 202599, y=y,x=x,h=h,w=w)
# (test_label, listImg)=u.getBalancedLabel(label_path, int(feature), True, startLine=182638, endLine=202599, startRow=0, numRow=0)
test_set =u.getImagesFromList(img_path, listImg, y=y,x=x,h=h,w=w, bw=bw)
if(bw==0):
    test_set=test_set.reshape(test_set.shape[0], test_set.shape[1], test_set.shape[2],1)

print("\n\nPercentuale di 1 nel test set: "+ str(u.CountFeatures(test_label)))

s = time.time()
print("Start time: "+str(s)+"\n\n")

test_loss, test_acc = model.evaluate(test_set, test_label)

e=time.time()
print("End time: "+str(e))
print ("Totaltime: "+str(e-s))

print("Test accuracy:"+str(test_acc))
print("Test loss:"+str(test_loss))
