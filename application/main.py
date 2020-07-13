import sys
from os.path import sep, expanduser, dirname
import os,glob
# Model imports 
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import keras.backend as K
import keras.utils.conv_utils as conv_utils
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate
from keras.engine.topology import Layer, InputSpec
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

from keras.models import load_model
from keras.utils import np_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing.image import image 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
import h5py
from BilinearUpSampling2D import BilinearUpSampling2D
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import math
import random
import cv2
from kivy.uix.rst import RstDocument



BATCH_SIZE = 32
IMG_HEIGHT = 256
IMG_WIDTH = 256
base_model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))

# Starting point for decoder
base_model_output_shape = base_model.layers[-1].output.shape #15, 20, 2048
# Starting number of decoder filters
decode_filters = int(int(base_model_output_shape[-1]) / 2)

# Define upsampling layer
def upproject(tensor, filters, name, concat_with):
    up_i = BilinearUpSampling2D((2, 2), name=name + '_upsampling2d')(tensor)
    up_i = Concatenate(name=name + '_concat')(
        [up_i, base_model.get_layer(concat_with).output])  # Skip connection
    up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convA')(up_i)
    up_i = LeakyReLU(alpha=0.2)(up_i)
    up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convB')(up_i)
    up_i = LeakyReLU(alpha=0.2)(up_i)
    return up_i

# Decoder Layers
decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape,
                  name='conv2')(base_model.output)

decoder = upproject(decoder, int(decode_filters / 2), 'up1', concat_with='res4a_branch2a') #30, 40, 256
decoder = upproject(decoder, int(decode_filters / 4), 'up2', concat_with='res3a_branch2a') # 60, 80, 128
decoder = upproject(decoder, int(decode_filters / 8), 'up3', concat_with='max_pooling2d_1') #120,  160, 64
decoder = upproject(decoder, int(decode_filters / 16), 'up4', concat_with='activation_1')  #240, 320, 64
decoder = upproject(decoder, int(decode_filters / 32), 'up5', concat_with='input_1')

# Extract depths (final layer)
conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

# Create the model
global model
model = Model(inputs=base_model.input, outputs=conv3)
model.load_weights('model_1.h5')

MAX_DEPTH = 120.0
MIN_DEPTH = 0.0

def DepthNorm(x, maxDepth):
    return maxDepth / x
    
def predict(model, images, minDepth=MIN_DEPTH, maxDepth=MAX_DEPTH, batch_size=8):
    # Support multiple RGBs, one RGB image, even grayscale 
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth
# Kivy import
from kivy.config import Config

# Setting graphics parameters in config.ini
Config.set('graphics','position','custom')
Config.set('graphics','top','200')
Config.set('graphics','left','400')
Config.set('graphics','resizable',False)
Config.set('graphics','width','1000')
Config.set('graphics','height','700')
Config.set('graphics','borderless',1)

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.garden.filebrowser import FileBrowser
from kivy.uix.label import Label
from kivy.core.text import Label as CoreLabel
from kivy.animation import Animation
from circular_progress_bar import CircularProgressBar
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.image import Image
from PIL import Image as PImage 
import PIL.ImageOps
from kivy.uix.settings import Settings,SettingsWithSidebar
from kivy.metrics import Metrics
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.graphics import *
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.rst import RstDocument
from kivy.uix.videoplayer import VideoPlayer
from resizeimage import resizeimage
import os

# Load the kv file for all the screens
Builder.load_file('capstone.kv')


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class Architecture(Screen):
    def on_enter(self):
        self.manager.current = 'Architecture'
class Requirements(Screen):
    def on_enter(self):
        self.manager.current = 'Requirements'

class Help(Screen):
    pass

class Logo(Screen):
    
    def skip(self, dt):
        self.manager.current = "Welcome"

    def on_enter(self, *args):
        Clock.schedule_once(self.skip, 10)

    def on_leave(self):
        Config.set('graphics','position','auto')
        from kivy.core.window import Window

class Welcome(Screen):
    # Simple animation to show the circular progress bar in action
    def animate(self, dt):
        bar = self.children[0]
        if bar.type == 'CircularProgressBar':
           if bar.value < bar.max:
               bar.value += 1
           else:
               self.manager.current = "Background_new"
               bar.value = bar.min

    # On start set the screen to Welcome
    def on_enter(self):
        l = Label(text='Welcome!',font_size= '20sp')
        self.add_widget(l)
        anim = Animation(opacity=0,duration=3)
        anim.start(l)
        Clock.schedule_once(self.skip,4)
    
    # Change the label on the screen
    def skip(self,dt):
        l = Label(text='                  ECE496 Monocular Depth Perception\nAuthors: Syed Shaher Raza, James Meijers, Sofian Zalouk',font_size ='20sp')
        self.add_widget(l)
        anim = Animation(opacity=0,duration=3)
        anim.start(l)
        Clock.schedule_once(self.load,4)

    # Load the progress bar and then the next screen
    def load(self,dt):
        center = Window.center
        progressbar = CircularProgressBar(pos=('400','250'))
        progressbar.font_size = '20sp'
        progressbar.label = CoreLabel('Loading our model!');
        progressbar.thickness = 5
        progressbar.progress_colour = (1,1,0,1)
        self.add_widget(progressbar)
        Clock.schedule_interval(self.animate, 0.25)

    # On leaving stop the progress bar animation
    def on_leave(self):
        Clock.unschedule(self.animate)

def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended

class Loading(Screen):
    pop_up = None 
    text_value = 0
    filename = ''
    def dismiss_popup(self):
        self.pop_up.dismiss()  
     
    def show_load(self):
        if self.text_value != 0:
           self.text_value = 0
           return
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self.pop_up = self._popup
        self.text_value = 1
        self._popup.open()

    def load(self, path, filename):
        filecur, file_extension = os.path.splitext(filename[0])
        self.ids['my_slider'].opacity = 0 
        self.ids['slider_label'].opacity = 0
        self.ids['slider_text'].opacity = 0 
        if file_extension != '.png' and file_extension != '.jpg':
           self.ids['needs_processing'].source = 'select.png' 
           self.ids['depth_map_final'].source = 'select.png' 
           self.ids['select_image'].opacity = 0
           self.ids['wrong_input'].opacity = 1 
           self.ids['depth_map_final'].opacity = 1
           self.ids['input_image'].text = '' 
           self.ids['select_depth_map'].opacity = 1
           self.dismiss_popup()
           self.filename = ''
           return

        im = PImage.open(filename[0])
        width, height = im.size
        if width != height or width <256 or height <256 or width > 1080 or height > 1080:
           self.ids['needs_processing'].source = 'select.png' 
           self.ids['depth_map_final'].source = 'select.png' 
           self.ids['select_image'].opacity = 0 
           self.ids['wrong_input'].opacity = 1 
           self.ids['input_image'].text = '' 
           self.ids['depth_map_final'].opacity = 1
           self.ids['select_depth_map'].opacity = 1
           self.dismiss_popup()
           self.filename = ''
           return

        if width != 256:
           with open(filename[0],'r+b') as f:
                with PImage.open(f) as image:
                     path = filename[0].split('.')
                     cover = resizeimage.resize_cover(image,[256,256])
                     cover.save(path[0]+'_resize.'+path[1],image.format)
           self.filename = path[0]+'_resize.'+path[1]
        else:
             self.filename = filename[0]
        	
        self.ids['input_image'].text = self.filename
        self.ids['depth_map_final'].opacity = 1 
        self.ids['depth_map_final'].source = 'select.png' 
        self.ids['needs_processing'].source = self.filename 
        self.ids['needs_processing'].opacity = 1
        self.ids['wrong_input'].opacity = 0 
        self.ids['select_image'].opacity = 0 
        self.ids['select_depth_map'].opacity = 1
        
        self.dismiss_popup()

    def change_image(self, *args):
        test_data = np.zeros((1,256,256,3))
        if self.filename == '':
           return 
        self.ids['depth_map_final'].opacity = 0
        self.ids['select_depth_map'].opacity = 0
        test_data[0] = plt.imread(self.filename)
        predictions = predict(model,test_data)
        #new_pred = predictions[:,:,:,0]
        #predictions = model.predict(test_data,batch_size=32)
        new_pred = predictions.reshape(predictions.shape[0],predictions.shape[1], predictions.shape[2])
        print("HRERE")
        plt.imsave('output.png',(new_pred[0]),cmap='gray')
        self.ids['depth_map_final'].source = 'output.png'
        self.ids['depth_map_final'].reload()
        self.ids['depth_map_final'].opacity = 100
        self.ids['my_slider'].opacity = 1
        self.ids['slider_label'].opacity = 1
        self.ids['slider_text'].opacity = 1
    def on_touch_move(self, *args):
        #print(self.ids['my_slider'].value)
        #img = cv2.imread(self.filename)
        #kernel = cv2.imread('output.png')[:,:,0]/255.0
        kernel = cv2.imread('output.png')[:,:,0]
        img = cv2.imread(self.filename)

        H,W = img.shape[:2]
        depth = cv2.bitwise_not(cv2.imread('output.png')[:,:,0])
        
        mask = np.zeros(depth.shape,np.uint8)
        TARGET = 255-self.ids['my_slider'].value 
        tolerance = 20
        for i in range(depth.shape[0]):
          for j in range(depth.shape[1]):
            if depth[i,j] < TARGET + tolerance and depth[i,j] > TARGET - tolerance:
              mask[i,j] = 0 
            else:
              mask[i,j] = 255
        
        mask = cv2.GaussianBlur(mask, (21,21),11 )
        blured = cv2.GaussianBlur(img, (21,21), 11)
        import scipy.ndimage as ndimage
        
        blurred_mask = ndimage.gaussian_filter(mask, sigma=(50, 50), order=0)
        #plt.imshow(blurred_mask)
        #plt.show()
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        (thresh, binRed) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
        #plt.imshow(opening)
        #plt.show()
        
        blended1 = alphaBlend(img, blured, opening)
        blended2 = alphaBlend(img, blured, 255- blurred_mask)
        blended3 = alphaBlend(img, blured, mask)
        blended4 = alphaBlend(img, blured, mask)
        
        #plt.imshow(blended4)

        xs = PImage.fromarray(blended1)
        xs.save('xc.png')
        self.ids['depth_map_final'].source = 'xc.png' 
        self.ids['depth_map_final'].reload()
        
class MyScreenManager(ScreenManager):
    pass

class Background(Screen):
    def on_enter(self):
        self.ids['document'].colors['paragraph'] = 'A8AAAEFF'
        self.ids['document'].text = text = """
\n\n Recovering depth from images is a computer vision problem that has important applications in autonomous driving , robotics, 3-D modeling, and physics. Depth estimation is essential to understanding a scene and the geometric relations present within it. Existing autonomous driving algorithms rely heavily on the use of sensors (such as LiDAR), which provide accurate depth estimates of the scene. Depth measuring sensors suffer from two major short-comings: they are very expensive, and suffer from low sampling rates. Due to their high cost, sensors can significantly increase the overall price of the autonomous driving system and due to their low sampling rate, they can significantly limit the effectiveness of autonomous driving systems, which must respond to real world input in a timely manner. Faced with these shortcomings, alternative methods for detecting depth are desirable to reduce costs and increase data acquisition rates.\n\n A natural candidate to replace sensors are image-based depth estimation methods. Much of the prior work in this area relies on stereopsis, which mimics human binocular vision through the use of two cameras capturing the same scene . Detecting depth from monocular images is a far more challenging task as it requires the use of more subtle cues such as motion parallax, perspective, object sizes, etc. . Furthermore, unlike their binocular counterparts, depth cannot be recovered deterministically from monocular images; many scenes could recreate the same image . However, the majority of scenes can be disregarded as physically implausible and depth can still be recovered with significant accuracy. In spite of these limitations, monocular depth detection remains highly desirable for various reasons. First, devices that produce monocular images are highly accessible and low-cost. In addition, monocular images are easily and abundantly found on the internet and through social media outlets. The abundance of potential data inputs inherently makes an accurate monocular depth perception method more far-reaching than its binocular counterpart.\n\n Given current depth detection systems, a gap can be identified in the industry for an inexpensive, easy-to-use, accurate, and accessible depth detection system. Sensors and binocular based systems each present a list of shortcomings that hinder their accessibility, and that can be overcome by an accurate monocular depth detection system. Through the use of machine learning models, accurate depth maps can be generated using inexpensive monocular systems. This would require an extensive exploration of state-of-the-art machine learning methods , combined with large training sets that must be gathered, processed, and refined. In addition, monocular depth cues can be processed and used as input parameters to the machine learning models to achieve higher accuracy and performance. All in all, a machine learning implementation provides the accurate and inexpensive monocular depth detection model that would fill the current gap in industry. """
        self.ids['document'].render()

class Background_new(Screen):
    def on_enter(self):
        self.ids['document'].colors['paragraph'] = 'A8AAAEFF'
        self.ids['document'].text = text = """
\n\n Recovering depth from images is a computer vision problem that has important applications in autonomous driving , robotics, 3-D modeling, and physics. Depth estimation is essential to understanding a scene and the geometric relations present within it. Existing autonomous driving algorithms rely heavily on the use of sensors (such as LiDAR), which provide accurate depth estimates of the scene. Depth measuring sensors suffer from two major short-comings: they are very expensive, and suffer from low sampling rates. Due to their high cost, sensors can significantly increase the overall price of the autonomous driving system and due to their low sampling rate, they can significantly limit the effectiveness of autonomous driving systems, which must respond to real world input in a timely manner. Faced with these shortcomings, alternative methods for detecting depth are desirable to reduce costs and increase data acquisition rates.\n\n A natural candidate to replace sensors are image-based depth estimation methods. Much of the prior work in this area relies on stereopsis, which mimics human binocular vision through the use of two cameras capturing the same scene . Detecting depth from monocular images is a far more challenging task as it requires the use of more subtle cues such as motion parallax, perspective, object sizes, etc. . Furthermore, unlike their binocular counterparts, depth cannot be recovered deterministically from monocular images; many scenes could recreate the same image . However, the majority of scenes can be disregarded as physically implausible and depth can still be recovered with significant accuracy. In spite of these limitations, monocular depth detection remains highly desirable for various reasons. First, devices that produce monocular images are highly accessible and low-cost. In addition, monocular images are easily and abundantly found on the internet and through social media outlets. The abundance of potential data inputs inherently makes an accurate monocular depth perception method more far-reaching than its binocular counterpart.\n\n Given current depth detection systems, a gap can be identified in the industry for an inexpensive, easy-to-use, accurate, and accessible depth detection system. Sensors and binocular based systems each present a list of shortcomings that hinder their accessibility, and that can be overcome by an accurate monocular depth detection system. Through the use of machine learning models, accurate depth maps can be generated using inexpensive monocular systems. This would require an extensive exploration of state-of-the-art machine learning methods , combined with large training sets that must be gathered, processed, and refined. In addition, monocular depth cues can be processed and used as input parameters to the machine learning models to achieve higher accuracy and performance. All in all, a machine learning implementation provides the accurate and inexpensive monocular depth detection model that would fill the current gap in industry. """
        self.ids['document'].render()

class Goal(Screen):
    def on_enter(self):
        self.ids['document'].colors['paragraph'] = 'A8AAAEFF'
        self.ids['document'].text = text = """
\n\n  The goal of this project is to develop a machine learning model to generate depth maps of scenes captured using monocular cameras that can match the accuracy of existing LIDAR equipment and state-of-the-art algorithms. """
        self.ids['document'].render()

class Goal_new(Screen):
    def on_enter(self):
        self.ids['document'].colors['paragraph'] = 'A8AAAEFF'
        self.ids['document'].text = text = """
\n\n The goal of this project is to develop a machine learning model to generate depth maps of scenes captured using monocular cameras that can match the accuracy of existing LIDAR equipment and state-of-the-art algorithms. """
        self.ids['document'].render()

class TestApp(App):
    my_manager = ObjectProperty(None)
    def build(self):
        self.icon = 'kivylogo1.jpeg'
        s = self.create_settings()
        my_manager = MyScreenManager()
        return my_manager 

if __name__ == "__main__":
    TestApp().run()
