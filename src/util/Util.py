import cv2
import numpy as np
from matplotlib import pyplot as plt
import re
import pickle

def getFilteredOneHotLabel(train_label_path, filterFeatures, valueFeatures, features, startLine=1, endLine=202599, startRow=0, numRow=0):
    assert 0 not in features
    file =  open(train_label_path,'r')
    lines = file.readlines()[startLine+1:endLine+2]

    vettore=[]
    startRow = startRow//len(features)
    for line in lines:
        words = re.split("  | |\n", line)
        # if(len(features)==1):
        #     vettore.append(int(words[features[0]]))
        # else:
        valoriDiFiltro=[int(words[filterFeatures[i]]) for i in range(len(filterFeatures))]
        if(valoriDiFiltro==valueFeatures):
            appoggio=[int(words[i])for i in features]
            if(appoggio.count(1)==1):
                vettore.append([appoggio.index(1), words[0]])

    vettore.sort(key=lambda x: x[0])

    labels=np.array([vettore[i][0] for i in range(len(vettore))])
    images=np.array([vettore[i][1] for i in range(len(vettore))])
    count=300000
    for i in range(len(features)):
        a=np.count_nonzero(labels==i)
        if(a<count):
            count=a

    print(count)
    if(startRow>=count):
        raise ValueError("startRow troppo grande")
    if(numRow == 0):
            numRow = (count-startRow)*len(features)

    index=[]
    num=0
    for i in range( len(features)) :
        num+=np.count_nonzero(labels==i-1)
        index.extend(list(range(startRow + num,  startRow + num+numRow//len(features))))
        if(labels[index[-1]]!=i):
            raise ValueError("Non ci sono abbastanza valori")

    images=images[index]
    labels=labels[index]

    shuffle = np.arange(labels.shape[0])
    np.random.shuffle(shuffle)
    labels= labels[shuffle]
    images = images[shuffle]

    return labels, images


def getOneHotLabel(train_label_path, features, startLine=1, endLine=202599, startRow=0, numRow=202598):
    assert 0 not in features
    file =  open(train_label_path,'r')
    lines = file.readlines()[startLine+1:endLine+2]
    # print(str(startLine+1) +": " + str(nLines+2))
    vettore=[]
    startRow = startRow//len(features)
    for line in lines:
        words = re.split("  | |\n", line)
        if(len(features)==1):
            vettore.append(int(words[features[0]]))
        else:
            appoggio=[int(words[i])for i in features]
            if(appoggio.count(1)==1):
                vettore.append([appoggio.index(1), words[0]])
    vettore.sort(key=lambda x: x[0])

    labels=np.array([vettore[i][0] for i in range(len(vettore))])
    images=np.array([vettore[i][1] for i in range(len(vettore))])
    count=300000
    for i in range(len(features)):
        a=np.count_nonzero(labels==i)
        if(a<count):
            count=a

    print(count)
    if(startRow>=count):
        raise ValueError("startRow troppo grande")
    if(numRow == 0):
            numRow = (count-startRow)*len(features)

    index=[]
    num=0
    for i in range( len(features)) :
        num+=np.count_nonzero(labels==i-1)
        index.extend(list(range(startRow + num,  startRow + num+numRow//len(features))))
        if(labels[index[-1]]!=i):
            raise ValueError("Non ci sono abbastanza valori")

    images=images[index]
    labels=labels[index]

    # print(index)
    # print(images)
    # print(labels)

    shuffle = np.arange(labels.shape[0])
    np.random.shuffle(shuffle)
    labels= labels[shuffle]
    images = images[shuffle]

    return labels, images


def getFilteredBalLabel(train_label_path, filterFeatures, valueFeatures , feature, isBool,startLine=1, endLine=202599, startRow=0, numRow=202598):
    assert feature!=0
    assert numRow%2==0
    file =  open(train_label_path,'r')
    lines = file.readlines()[startLine+1:endLine+2]
    vettore=[]
    startRow = startRow//2
    #print(valueFeatures)
    for line in lines:
        words = re.split("  | |\n", line)
        appoggio=[int(words[filterFeatures[i]]) for i in range(len(filterFeatures))]
        #print(appoggio)
        if(appoggio==valueFeatures):
            #print("*"+ words[0])
            vettore.append([int(words[feature]),words[0]])
    vettore.sort(key=lambda x: x[0])

    labels=np.array([vettore[i][0] for i in range(len(vettore))])
    images=np.array([vettore[i][1] for i in range(len(vettore))])
    count=np.count_nonzero(labels==-1)
    print(count)
    if(numRow == 0):
        if(count > len(labels)//2):
            numRow = (len(labels)-count-startRow)*2
        else:
            numRow = (count-startRow)*2

    index=list(range(startRow,startRow +numRow//2))
    if(index[-1]>=count):
        raise ValueError("Non ci sono abbastanza valori")
    index.extend(list(range(count+startRow,count+startRow+numRow//2)))
    images=images[index]
    labels=labels[index]

    shuffle = np.arange(labels.shape[0])
    np.random.shuffle(shuffle)
    labels= labels[shuffle]
    images = images[shuffle]

    if(isBool):
          labels[labels == -1] = 0
    return labels, images

def getBalancedLabel(train_label_path, feature, isBool,startLine=1, endLine=202599, startRow=0, numRow=202598):
    assert feature!=0
    assert numRow%2==0
    file =  open(train_label_path,'r')
    lines = file.readlines()[startLine+1:endLine+2]
    vettore=[]
    startRow = startRow//2
    for line in lines:
        words = re.split("  | |\n", line)
        vettore.append([int(words[feature]),words[0]])
    vettore.sort(key=lambda x: x[0])

    labels=np.array([vettore[i][0] for i in range(len(vettore))])
    images=np.array([vettore[i][1] for i in range(len(vettore))])
    count=np.count_nonzero(labels==-1)
    print(count)
    if(numRow == 0):
        if(count > len(labels)//2):
            numRow = (len(labels)-count-startRow)*2
        else:
            numRow = (count-startRow)*2

    index=list(range(startRow,startRow +numRow//2))
    if(index[-1]>=count):
        raise ValueError("Non ci sono abbastanza valori")
    index.extend(list(range(count+startRow,count+startRow+numRow//2)))
    images=images[index]
    labels=labels[index]

    # print(index)
    # print(images)
    # print(labels)

    shuffle = np.arange(labels.shape[0])
    np.random.shuffle(shuffle)
    labels= labels[shuffle]
    images = images[shuffle]

    if(isBool):
          labels[labels == -1] = 0
    return labels, images

def getLabels(train_label_path, features, isBool,startLine=1, nLines=202599):
    assert 0 not in features
    file =  open(train_label_path,'r')
    lines = file.readlines()[startLine+1:nLines+2]
    # print(str(startLine+1) +": " + str(nLines+2))
    vettore=[]
    for line in lines:
        words = re.split("  | |\n", line)
        if(len(features)==1):
            vettore.append(int(words[features[0]]))
        else:
            vettore.append([int(words[i])for i in features])
    vettore=np.asarray(vettore)
    if(isBool):
         vettore[vettore == -1] = 0
    return vettore


def getImagesFromList(train_image_path, listImg, y=0,x=0,h=218,w=178, bw=1):
    train_set = np.array([cv2.imread((train_image_path+'/'+listImg[i]),bw)[y:y+h, x:x+w] for i in range(len(listImg))])
    train_set = train_set.astype('float32') / 255
    return train_set


def getImages(train_image_path,  startImg=1, nImg=202599, y=0,x=0,h=218,w=178,bw=1):

    train_set = np.array([cv2.imread((train_image_path+'/{:0>6}.jpg').format(i),1)[y:y+h, x:x+w] for i in range(startImg,nImg+1)])
    train_set = train_set.astype('float32') / 255
    return train_set

def printResult(history, needSave=False, path=None, fileName=None):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()

    f, axarr = plt.subplots(1,2, sharey= True)
    axarr[0].plot(epochs, acc, 'bo', label='Training acc')
    axarr[0].plot(epochs, val_acc, 'b', label='Validation acc')
    axarr[0].set_title('Training and validation accuracy')
    axarr[0].legend()

    axarr[1].plot(epochs, loss, 'bo', label='Training loss')
    axarr[1].plot(epochs, val_loss, 'b', label='Validation loss')
    axarr[1].set_title('Training and validation loss')
    axarr[1].legend()

    f.set_size_inches(20, 10)
    if(needSave):
        f.savefig(path+"/"+fileName+".png")
        with open(fileName+'.pickle', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    plt.show()

    # f = plt.figure()
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()



def CountFeatures(vettore):
    return (np.count_nonzero(vettore == 1))/len(vettore)


# model.save('cats_and_dogs_small_2.h5') per salvare i risultati

def historyUnion(lista,graph_path, fileName):
    lista=[ pickle.load(open(l, 'rb')) for l in lista]
    keys=list(lista[0].keys())
    newHist={k:[] for k in keys}
    for k in keys:
        for l in lista:
            newHist[k]=newHist[k]+l[k]


    printResult( type('Hist', (object,), {'history':newHist})(),True,graph_path, fileName)



# ======================= MAIN ==================================
# label_path="../CelebA/Anno/list_attr_celeba.txt"
#getFilteredBalLabel(label_path,[1,2], [-1,1], 3, True, startLine=1, endLine=30,startRow=0, numRow=4)



#prende le prime 21 img e label associate
# trai = getImages("../CelebA/Img/img_align_celeba",29,30, y=0,x=0,h=105,w=178)
# #lab = getLabels("/home/psico/Scrivania/Università/Biometria/WorkspacePython/Progetto/CelebA/Anno/list_attr_celeba.txt",[1,2,3,4,5,6,7,8,9],True,nLines=21)
# #
# # print(lab[20])
# cv2.imshow('faces',trai[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
