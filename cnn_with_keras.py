from keras.layers import Input,Conv2D,BatchNormalization,MaxPooling2D,Dropout,Flatten,Dense
from keras.models import Model
from keras.utils import Sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
import numpy as np
from PIL import Image
import os
import random
import time

PIC_SHAPE = (56,176,3)
CHARSET = '0123456789abcdefghijklmnopqrstuvwxyz'
CAPTCHA_LEN = 4

TRAIN_IMG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),'pic','train')
MODEL_SAVE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),'module')

# Model
inputs = Input(PIC_SHAPE)
outputs = inputs
outputs = Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu')(outputs)
outputs = Conv2D(filters=32,kernel_size=(3,3),activation='relu')(outputs)
outputs = BatchNormalization()(outputs)
outputs = MaxPooling2D(pool_size=(2,2))(outputs)
outputs = Dropout(0.3)(outputs)
outputs = Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu')(outputs)
outputs = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(outputs)
outputs = BatchNormalization()(outputs)
outputs = MaxPooling2D(pool_size=(2,2))(outputs)
outputs = Dropout(0.3)(outputs)
outputs = Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu')(outputs)
outputs = Conv2D(filters=128,kernel_size=(3,3),activation='relu')(outputs)
outputs = BatchNormalization()(outputs)
outputs = MaxPooling2D(pool_size=(2,2))(outputs)
outputs = Dropout(0.3)(outputs)
outputs = Conv2D(filters=256,kernel_size=(3,3),activation='relu')(outputs)
outputs = BatchNormalization()(outputs)
outputs = MaxPooling2D(pool_size=(2,2))(outputs)
outputs = Flatten()(outputs)
outputs = Dropout(0.3)(outputs)
outputs = [Dense(len(CHARSET),activation='softmax')(outputs),
          Dense(len(CHARSET),activation='softmax')(outputs),
          Dense(len(CHARSET),activation='softmax')(outputs),
          Dense(len(CHARSET),activation='softmax')(outputs),
          ]
model = Model(inputs=inputs,outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


def categorical(captcha_text):
    """
    :param captcha_text:
    :return:
    """
    num_value = [CHARSET.find(one_char) for one_char in captcha_text]
    return to_categorical(num_value,num_classes=len(CHARSET),dtype='int32')

def batch_generator(data_dict,batch_size=16):
    """
    这是一个自定义的数据生成器写法，推荐是使用后面那个继承keras.utils.Sequence的写法。
    注意总样本数/Batch_size 必须能整除，不然最后一个批次数据的大小跟前面的不一致，从而引发数量不匹配的问题
    TODO:可以考虑加入不整除的丢弃余留的样本。\
    TODO:加入shuffle机制
    :param data_dict:   {'captcha_filename':'value'...}
    :param batch_size:
    :return:  One Batch data
    """
    pic_shape = (60,200,3)
    key_list = [ k for k in data_dict.keys()]
    print(len(key_list))
    steps = 0
    while True:
        x = np.empty((batch_size,pic_shape[0],pic_shape[1],pic_shape[2]))
        y = [[] for _ in range(6)]
        filelist = key_list[steps:steps+batch_size]
        for i,filename in enumerate(filelist):
            x[i] = np.array(Image.open("./data/6_imitate_vali_set/" + filename + ".jpg")) / 255.0
            for j in range(6):
                y[j].append(categorical(data_dict[filename])[j])

        y = [digit for digit in np.asarray(y)]
        steps = steps + batch_size

        # 这里需要有索引跳回开头得机制
        # 不然1个epoch结束后，就会超过索引了
        if steps == len(key_list):
            steps = 0
        yield x, y

class DataGenerator(Sequence):
    """
    继承keras.utils.Sequence，在model.fit_generator时不需要指定step
    来源: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    略有修改
    """
    def __init__(self, train_data, batch_size=32, dim=(32,32,32), n_channels=1,
                 charset_len=36, char_nums=6,shuffle=True,pic_path=TRAIN_IMG_PATH):
        self.data = train_data
        self.ids = [k for k in self.data.keys()]
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.charset_len = charset_len
        self.char_nums = char_nums
        self.shuffle = shuffle
        self.pic_path = pic_path
        self.on_epoch_end()

    def __len__(self):
        """
        Notice:
        It keep the part of data divisible by batch_size,
        the other part would not use for train
        :return:
        """
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index:
        :return:
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ids_temp = [self.ids[k] for k in indexes]
        x, y = self.__data_generation(ids_temp)
        return (x, y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_temp):
        """
        Generates data containing batch_size samples
        :param ids_temp:
        :return:
        """
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = [ [] for _ in range(self.char_nums) ]
        for i, filename in enumerate(ids_temp):
            x[i] = np.array(Image.open(self.pic_path + os.sep + filename + ".jpg"))/255.0
            cate_y = categorical(filename)
            for index in range(self.char_nums):
                y[index].append(cate_y[index])
        return x, [one_char for one_char in np.asarray(y)]

# train_data 是一个字典，记录文件名及对应的label值
# 这里因为生成的数据文件名就是label值，此处显得有些多余，但有些情况生成的数据并非如此
train_data = {}
for filename in os.listdir(TRAIN_IMG_PATH):
    train_data[filename.split('.')[0]] = filename.split('.')[0]
train_data_generator = DataGenerator(train_data,batch_size=400,dim=(PIC_SHAPE[0],PIC_SHAPE[1]),n_channels=PIC_SHAPE[2],
                                     charset_len=len(CHARSET),char_nums=CAPTCHA_LEN,pic_path=TRAIN_IMG_PATH)

# 验证集从训练集中拉取一部分
CHOSE_PER = 0.7
validate_data = {}
random.seed(time.time())
select_list = [k for k in train_data.keys()]
random.shuffle(select_list)
for filename in select_list[:int(len(select_list)*CHOSE_PER)]:
    validate_data[filename] = train_data[filename]
validate_data_generator = DataGenerator(train_data,batch_size=600,dim=(PIC_SHAPE[0],PIC_SHAPE[1]),n_channels=PIC_SHAPE[2],
                                     charset_len=len(CHARSET),char_nums=CAPTCHA_LEN,pic_path=TRAIN_IMG_PATH)

checkpoint = ModelCheckpoint(MODEL_SAVE_PATH + os.sep + 'new_module.h5', monitor='val_dense_4_acc', verbose=1, save_best_only=True, mode='auto')
earlystop = EarlyStopping(monitor='val_dense_4_acc', patience=10, verbose=1, mode='auto')
# change histogram_freq to 0 , https://github.com/aurora95/Keras-FCN/issues/50
# tensorBoard = TensorBoard(log_dir = "./logs", histogram_freq = 0)
callbacks_list = [checkpoint,earlystop]
model.fit_generator(generator=train_data_generator,validation_data=validate_data_generator,max_queue_size=125,
                    use_multiprocessing=True,verbose=1,epochs=100,callbacks=callbacks_list)