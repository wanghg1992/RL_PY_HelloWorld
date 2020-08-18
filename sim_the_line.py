from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    # outputs = 1
    # return outputs\
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    return Wx_plus_b

x_data = np.linspace(-1, 2, 400)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#define the placeholder
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
#add layer1
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
#add layer2
prediction=add_layer(l1,10,1,activation_function=None)

#calculate error
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

#optimization
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# print("noise:\n", noise.shape)
# print("x_data:\n", x_data.shape)

#important step:init
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
plt.pause(1)

for i in range(1000):
    #train
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})

    if i%10==0:
        # attempt to remove
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value=sess.run(prediction,feed_dict={xs:x_data})
        lines=ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)
