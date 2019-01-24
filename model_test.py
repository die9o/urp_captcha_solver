import keras
from keras.models import load_model
from keras.backend import clear_session
import os
import numpy as np
from PIL import Image
import random,time
import requests
import uuid

MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),'module')
TEST_IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'pic','test')

CHARSET = '0123456789abcdefghijklmnopqrstuvwxyz'
CAPTCHA_LEN =4
FILE_EXT = '.jpg'


def download_img(num=1):
    """
    从网站上下载验证码图片
    :param num:
    :return:
    """
    print('正在下载图片保存到pic/test:')
    for i in range(num):
        r = requests.get('http://jwxt.imu.edu.cn/img/captcha.jpg',timeout=10)
        with open(TEST_IMG_PATH + os.sep + '{0}.jpg'.format(uuid.uuid4()),'wb') as f:
            f.write(r.content)
        print('\r 下载进度：{0}/{1}'.format(i+1,num),end='')


def _resize_img(_file):
    img = Image.open(_file)
    w,h = img.size
    new_img = img.crop([2,2,w-2,h-2])
    new_img.save(_file)


def resize_all_imgs():
    for _file in os.listdir(TEST_IMG_PATH):
        _file = TEST_IMG_PATH + os.sep + _file
        _resize_img(_file)


def model_test(num=10):
    """
    验证原来项目设计的模型
    :return:
    """
    print('清空原文件夹图片 pic/test')
    for img in os.listdir(TEST_IMG_PATH):
        os.remove(TEST_IMG_PATH + os.sep + img)
    download_img(num)
    resize_all_imgs()

    model6 = load_model(MODEL_PATH + os.sep + 'new_module.h5')
    chose_percentage = 1.0
    validate_data = {}
    random.seed(time.time())
    select_list = [k.split('.')[0] for k in os.listdir(TEST_IMG_PATH)]
    random.shuffle(select_list)
    filename_list  = select_list[:int(len(select_list) * chose_percentage)]
    test_data = np.stack([np.array(Image.open(TEST_IMG_PATH + os.sep + img_file +FILE_EXT))/255.0
                          for img_file in filename_list])
    prediction = model6.predict(test_data)
    ## 这里设计的模型，得到的预测是一个长度4的数组[[0],[1],[2],[3]]
    ##  [0] ：表示第一个数字的预测结果集，每个预测结果为一个数组，长度为36，即[ [len(36)],[len(36)]...]
    ##  对于一个字符的预测结果，因为模型执行过BatchNormalization
    ##  长度36的数组并不是类似的categorical的二进制矩阵结果[0 0 1 0 0 ...]，
    ##  它是方差接近1的一个矩阵，所以要找出对应的1可以通过np.argmax去获取下标

    predit_list = [[] for _ in range(CAPTCHA_LEN)]
    for i in range(CAPTCHA_LEN):
        for one_char in prediction[i]:
            predit_list[i].append(CHARSET[np.argmax(one_char)])

    for i in range(len(filename_list)):
        predict_val = ''.join([predit_list[x][i] for x in range(CAPTCHA_LEN)])
        origin_file = TEST_IMG_PATH + os.sep + filename_list[i] + FILE_EXT
        os.rename(origin_file,TEST_IMG_PATH + os.sep + predict_val + FILE_EXT)

    # predict_correct = 0
    # for i in range(len(filename_list)):
    #     predict_val = ''.join([predit_list[x][i] for x in range(CAPTCHA_LEN)])
    #     if predict_val == filename_list[i].split('_')[0]:
    #         predict_correct += 1
    # print('预测结果: {0}/{1}'.format(predict_correct,len(filename_list)))
    clear_session()
    print('识别结束，请到目录pic/test下检查结果')

    
model_test(10)