import Util as u
import numpy as np
from keras.utils import Sequence
# Here, `img_set` is list of path to the images
# and `label_set` are the associated classes.

class DataSequence(Sequence):

    def __init__(self, img_set, label_set, batch_size, train_image_path,y,x,h,w,bw):
        self.img_set, self.label_set = img_set, label_set
        self.batch_size = batch_size
        self.train_image_path=train_image_path
        self.y, self.x, self.h, self.w=y,x,h,w
        self.bw=bw

    def __len__(self):
        return int(np.ceil(len(self.img_set) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.img_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.label_set[idx * self.batch_size:(idx + 1) * self.batch_size]

        set=u.getImagesFromList(self.train_image_path, batch_x, y=self.y, x=self.x, h=self.h, w=self.w, bw=self.bw)
        if(self.bw==0):
            set=set.reshape(set.shape[0], set.shape[1], set.shape[2],1)
        return set, batch_y
