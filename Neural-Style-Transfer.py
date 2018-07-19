
# coding: utf-8

# In[251]:


import numpy as np
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img


# In[252]:


style_image_path = 'style.jpeg'
content_image_path = 'content.jpeg'

target_dim = 350

style_image_original = Image.open(style_image_path)
content_image_original = Image.open(content_image_path)
content_image = content_image_original.resize((target_dim, target_dim))
style_image = style_image_original.resize((target_dim, target_dim))


# In[253]:


content_array = np.asarray(content_image, dtype = 'float32')
style_array = np.asarray(style_image, dtype = 'float32')
content_array = np.expand_dims(content_array, axis = 0)
style_array = np.expand_dims(style_array, axis = 0)


# In[254]:


content_array[:, :, :, 0] -= 103.939
content_array[:, :, :, 1] -= 116.779
content_array[:, :, :, 2] -= 123.68
content_array = content_array[:, :, :, ::-1]

style_array[:, :, :, 0] -= 103.939
style_array[:, :, :, 1] -= 116.779
style_array[:, :, :, 2] -= 123.68
style_array = style_array[:, :, :, ::-1]


# In[255]:


content_image = K.variable(content_array)
style_image = K.variable(style_array)
combination_image = K.placeholder((1, target_dim, target_dim, 3))

print(style_array.shape)
print(content_array.shape)


# In[256]:


input_tensor = K.concatenate([content_image, style_image, combination_image], axis = 0)


# In[257]:


model = VGG16(input_tensor = input_tensor, weights = 'imagenet', include_top = False)


# In[258]:


content_weight = .075
style_weight = 1.0
total_variation_weight = 1.0

loss = K.variable(0.)

layers = dict([(layer.name, layer.output) for layer in model.layers])
layers


# In[259]:


def content_loss(content, combination):
    return K.sum(K.square(content - combination))

layer_features = layers['block4_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight*content_loss(content_image_features, combination_features)


# In[260]:


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features))


# In[261]:


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = target_dim*target_dim
    return K.sum(K.square(S - C))/ (4.*(channels ** 2) * (size ** 2))


# In[262]:


feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']


# In[263]:


for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight/len(feature_layers)) * sl


# In[264]:


#Regulariztion
height = target_dim
width = target_dim
def total_variation_loss(x):
    a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

loss += total_variation_weight * total_variation_loss(combination_image)


# In[265]:


grads = K.gradients(loss, combination_image)

outputs = [loss]
outputs += grads
f_outputs = K.function([combination_image], outputs)
def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_values = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_values, grad_values


# In[266]:


class Evaluator(object):
    
    def __init__(self):
        self.loss_values = None
        self.grad_values = None
    
    def loss(self, x):
        assert self.loss_values is None
        loss_values, grad_values = eval_loss_and_grads(x)
        self.loss_values = loss_values
        self.grad_values = grad_values
        return self.loss_values

    def grads(self, x):
        assert self.loss_values is not None
        grad_values = np.copy(self.grad_values)
        self.loss_values = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()


# In[ ]:


import time
from scipy.optimize import fmin_l_bfgs_b

x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

iterations = 25

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))


# In[ ]:


from scipy.misc import imsave
x = x.reshape((height, width, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')

img = Image.fromarray(x, 'RGB')
img.save('image.jpeg')

