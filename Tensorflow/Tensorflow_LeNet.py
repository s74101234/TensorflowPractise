from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

#讀取圖片
def read_img(path,img_height,img_width):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
#            img = mpimg.imread(im)
            img_resize=transform.resize(img,(img_height,img_width))
            imgs.append(img_resize)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

#batch function
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
        
def buildLeNetModel(input_tensor,img_channl,img_height,img_width,num_classes):
    nodes_width,nodes_height,nodes_channl=img_width,img_height,img_channl
    #conv1
    conv1 = tf.layers.conv2d(
      inputs=input_tensor,
      filters=20,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #pool1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    nodes_width /=2;nodes_height /=2;nodes_channl = 20
    #conv2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=50,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #pool2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    nodes_width /=2;nodes_height /=2;nodes_channl = 50
        
    nodes = int(nodes_width * nodes_height * nodes_channl)
    reshaped = tf.reshape(pool2,[-1,nodes])
    #fc1
    fc1 = tf.layers.dense(inputs=reshaped, 
                  units=500, 
                  activation=tf.nn.relu,
                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    #fc2
    logit = tf.layers.dense(inputs=fc1, 
                  units=num_classes, 
                  activation=None,
                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    return logit

def saveTrainModels(img_channl,img_height,img_width,num_classes,
                    saveModelPath,epochs,batch_size,x_train,y_train,x_test,y_test):
    x=tf.placeholder(tf.float32,shape=[None,img_width,img_height,img_channl])
    y_=tf.placeholder(tf.int32,shape=[None,])
    logits = buildLeNetModel(x,img_channl,img_height,img_width,num_classes) 
    
    loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
    train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
    acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     
    config = tf.ConfigProto(
            log_device_placement=True,
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0,
            allow_soft_placement=True)
    
    sess = tf.Session(config=config)
    saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print("epoch: %d" %epoch)
        #訓練
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err; train_acc += ac; n_batch += 1
        print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
        print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
        
        #驗證
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_test, y_test, batch_size, shuffle=False):
            err, ac = sess.run([loss,acc], feed_dict={x: x_test, y_: y_test})
            val_loss += err; val_acc += ac; n_batch += 1
        print("   validation loss: %f" % (val_loss/ n_batch))
        print("   validation acc: %f" % (val_acc/ n_batch))
    
    saver.save(sess,saveModelPath)
    sess.close()
    
if __name__ == "__main__":
    #參數設定
    img_height, img_width, img_channl = 200, 200 , 3
    num_classes = 5
    batch_size = 20
    epochs = 50
    dataSplitRatio=0.8
    readDataPath = "./../trainData/fruits/"
    saveModelPath = "./trainModels/Tensorflow_LeNet.ckpt"
    
    #載入資料
    data,label = read_img(readDataPath,img_height,img_width)
    
    #順序隨機
    num_example=data.shape[0]
    arr=np.arange(num_example)
    np.random.shuffle(arr)
    data=data[arr]
    label=label[arr]
    
    #切割資料
    #num_example=data.shape[0]
    s=np.int(num_example*dataSplitRatio)
    x_train=data[:s]
    y_train=label[:s]
    x_val=data[s:]
    y_val=label[s:]
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    
    #訓練及保存模型
    saveTrainModels(img_channl,img_height,img_width,num_classes,
                    saveModelPath,epochs,batch_size,x_train,y_train,x_val,y_val)