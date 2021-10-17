import numpy as np
import struct
import pickle
import matplotlib.pyplot as plt
from PIL import Image

#data load
def load_mnist(is_training=True):
    if is_training:
        fd = open('./Mnist/train-images.idx3-ubyte','rb')
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 784)).astype(np.float32)

        fd = open('./Mnist/train-labels.idx1-ubyte','rb')
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.int32)

        #trX = trainX / 255.
        trX = trainX

        return trX, trY
    else:
        fd = open('./Mnist/t10k-images.idx3-ubyte','rb')
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 784)).astype(np.float)

        fd = open('./Mnist/t10k-labels.idx1-ubyte','rb')
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        #teX = teX / 255.
        teX = teX

        return teX, teY


def pca(data_mat, top_n_feat=99999999):
    num_data,dim = data_mat.shape

    mean_vals = data_mat.mean(axis=0)# データ中心化
    mean_removed = data_mat - mean_vals

    cov_mat = np.cov(mean_removed, rowvar=0)#共分散行列（Find covariance matrix）

    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))# 固有値(Find eigenvalues and eigenvectors)
    #print(eig_vals)
    #print(eig_vects)
    eig_val_index = np.argsort(eig_vals) # 固有値を大きい順にソート
    eig_val_index = eig_val_index[:-(top_n_feat + 1) : -1]# 大きい方からtop_n_feat個の固有値
    reg_eig_vects = eig_vects[:, eig_val_index]# 上記固有値に対応する固有ベクトル
    low_d_data_mat = mean_removed * reg_eig_vects# 上記ベクトルが張る空間へ射影
    new_vecte = reg_eig_vects.T * reg_eig_vects
    # top_n_feat個の固有ベクトル
    print(new_vecte)
    # low_d_data_mat を原空間の座標に変換
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals
    return low_d_data_mat, recon_mat

#img　画像を表示用に1つにまとめる
def comb_imgs(origin_imgs, col, row, each_width, each_height, new_type):
    new_img = Image.new(new_type, (col* each_width, row* each_height))
    for i in range(len(origin_imgs)):
        each_img = array_to_img(np.array(origin_imgs[i]).reshape(each_width, each_width))
        new_img.paste(each_img, ((i % col) * each_width, (i // col) * each_width))
    return new_img
# 配列を画像に
def array_to_img(array):
    #array=array*255
    new_img=Image.fromarray(array.astype(np.uint8))
    return new_img

def cal_CCR(data_mat, top_n_feat=99999999):
    num_data,dim = data_mat.shape
    mean_vals = data_mat.mean(axis=0)# データ中心化
    mean_removed = data_mat - mean_vals
    cov_mat = np.cov(mean_removed, rowvar=0)#共分散行列（Find covariance matrix）
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))# 固有値(Find eigenvalues and eigenvectors)
    eig_val_index = np.argsort(eig_vals) # 固有値を大きい順にソート
    list = []
    total = 0
    for i in range(top_n_feat):
        total += eig_vals[i] / eig_vals.sum()
        list.append(total)
    return list

def QP(data_mat):
    dimention = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 764]
    list = []
    for i in dimention:
        num_data, i = data_mat.shape

        mean_vals = data_mat.mean(axis=0)# データ中心化
        mean_removed = data_mat - mean_vals

        cov_mat = np.cov(mean_removed, rowvar=0)#共分散行列（Find covariance matrix）

        eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))# 固有値(Find eigenvalues and eigenvectors)
        #print(eig_vals)
        #print(eig_vects)
        eig_val_index = np.argsort(eig_vals) # 固有値を大きい順にソート
        eig_val_index = eig_val_index[:-(i + 1) : -1]# 大きい方からtop_n_feat個の固有値
        reg_eig_vects = eig_vects[:, eig_val_index]# 上記固有値に対応する固有ベクトル
        low_d_data_mat = mean_removed * reg_eig_vects# 上記ベクトルが張る空間へ射影
        new_vecte = reg_eig_vects.T * reg_eig_vects
        # top_n_feat個の固有ベクトル
        # low_d_data_mat を原空間の座標に変換
        recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals
        qp = np.linalg.norm(recon_mat - data_mat)
        list.append(qp)
    return list

X_train, y_train= load_mnist(is_training=True)
X_test, y_test= load_mnist(is_training=False)
low_train_img = comb_imgs(X_train, 100, 100, 28, 28, 'L')

# train data すべてを表示 
# low_train_img.show()

#8_img 「8」だけを抜き出して、origin_8_imgsに格納、表示
origin_4_imgs = []
for i in range(10000):
     if y_train[i] == 4 and len(origin_4_imgs) < 10000:
        origin_4_imgs.append(X_train[i])
low_4_img = comb_imgs(origin_4_imgs, 25, 40, 28, 28, 'L')
#low_4_img.show()

# origin_8_imgsに対してPCAをかけ、pcadim 次元の空間を構成
# origin_8_imgs をその空間に対して射影した画像を表示
#pcadim = 1
#while pcadim <= 764:
#   pcadim *= 2
#   low_d_feat_for_8_imgs, recon_mat_for_8_imgs = pca(np.array(origin_8_imgs), pcadim)
#  low_d_img = comb_imgs(recon_mat_for_8_imgs, 25, 40, 28, 28, 'L')
#  low_d_img.show(title=pcadim)

list_4 = cal_CCR(np.array(origin_4_imgs), 764)
pcadims = [1, 2, 4, 8, 16, 32, 64, 128, 256, 300, 350, 400, 450, 512, 764]
for i in pcadims:
    
    print(str(i) + ":" + str(list_4[i - 1]))

#plt.plot([n for n in range(764)], list_4, color = "red")
#plt.show()


# origin_0_imgsに対してPCAをかけ、pcadim 次元の空間を構成
# origin_0_imgs をその空間に対して射影した画像を表示
#pcadim = 3
#low_d_feat_for_0_imgs, recon_mat_for_0_imgs = pca(np.array(origin_0_imgs), pcadim)
#low_d_img = comb_imgs(recon_mat_for_0_imgs, 25, 40, 28, 28, 'L')
#low_d_img.show()

# origin_8_imgsに対してPCAをかけ、pcadim 次元の空間を構成
# origin_0_imgsをその空間に対して射影した画像を表示O
#list_4 = cal_CCR(np.array(origin_4_imgs), 764)
#pcadims = [1, 2, 4, 8, 16, 32, 64, 128, 256, 300, 350, 400, 450, 512, 764]

#for i in range(15):
    #low_d_feat_for_4_imgs, recon_mat_for_4_imgs = pca(np.array(origin_4_imgs), pcadims[i])
    #low_d_img = comb_imgs(recon_mat_for_4_imgs, 25, 40, 28, 28, 'L')
    #low_d_img.show()
#    print(list_4[i])
