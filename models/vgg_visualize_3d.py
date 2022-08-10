#import cv2



# Install mayavi and vtk from https://www.lfd.uci.edu/~gohlke/pythonlibs/#vtk
# pip install "wheels package"

import numpy as np
from scipy import ndimage

from skimage.filters import sobel_h
from skimage.filters import sobel_v

#from sa_decomp_layer import SADecompLayer

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #disables GPU 
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

#tf.__version__
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

plt.rcParams['figure.figsize'] = [10,10]

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications import VGG16

from mayavi  import mlab 




model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

def get_filter(layer):
    layer = model.layers[layer]

    # check for convolutional layer
    if 'conv' not in layer.name:
        return None
    # get filter weights
    filters, biases = layer.get_weights()
    return (filters)
    #print(layer.name, filters.shape)




def getSobelAngle(f):

    s_h = sobel_h(f)
    s_v = sobel_v(f)

    return(np.arctan2(s_h,s_v)+(np.pi/2))

def getSymAntiSym(filter):

    #patches = extract_image_patches(filters, [1, k, k, 1],  [1, k, k, 1], rates = [1,1,1,1] , padding = 'VALID')
    #print(patches)
    mat_flip_x = np.fliplr(filter)

    mat_flip_y = np.flipud(filter)

    mat_flip_xy =  np.fliplr( np.flipud(filter))

    sum = filter + mat_flip_x + mat_flip_y + mat_flip_xy
    mat_sum_rot_90 = np.rot90(sum)
    
    return  (sum + mat_sum_rot_90) / 8, filter - ((sum + mat_sum_rot_90) / 8)


def normalizeToOne(mat):
    return mat/np.linalg.norm(mat)


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


'''
Display Functions
'''
def onpick(event):
    thisline = event.artist
    #xdata = thisline.get_xdata()
    #ydata = thisline.get_ydata()
    #ind = event.ind
    #points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', thisline.get_gid())



if __name__ == "__main__":

    filters = get_filter(1)

    fig = plt.figure()

    ax = plt.axes(projection='3d')

    for i in range(filters.shape[-1]):
        filter = filters[:,:,:, i]
        #for j in range(filter.shape[-1]):
        f = filter[:,:, 1]
        # Data for a three-dimensional line
        '''zline = np.linspace(0, 15, 1000)
        xline = np.sin(zline)
        yline = np.cos(zline)
        ax.plot3D(xline, yline, zline, 'gray')'''

        sym, anti = getSymAntiSym(normalizeToOne(f))
        sym_mag = np.linalg.norm(sym)
        anti_mag = np.linalg.norm(anti)

        theta = getSobelAngle(f)
        theta = theta[theta.shape[0]//2, theta.shape[1]//2]

        # Data for three-dimensional scattered points
        zdata = sym_mag
        xdata = anti_mag*np.sin(theta)
        ydata = anti_mag*np.cos(theta) 

        print(zdata.shape, xdata.shape, ydata.shape)
        mlab.points3d(xdata, ydata, zdata,  colormap='Spectral', scale_factor=.1)    
    
    #https://stackoverflow.com/questions/24308049/how-to-plot-proper-3d-axes-in-mayavi-like-those-found-in-matplotlib
    lensoffset = 0
    xx = yy = zz = np.arange(-1.5,1.5,0.1)
    xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
    mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=0.01)
    mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=0.01)
    mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=0.01)
    mlab.show()
