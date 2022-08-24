#import cv2



# Install mayavi and vtk from https://www.lfd.uci.edu/~gohlke/pythonlibs/#vtk
# pip install "wheels package"

from msilib.schema import Error
from turtle import color
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

LAYER = 15
FILTER = 400
CHANNEL = 300


RGB = ['R','G','B']

model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

def get_filter(layer):
    layer = model.layers[layer]

    # check for convolutional layer
    if 'conv' not in layer.name:
        raise ValueError('Layer must be a conv. layer')
    # get filter weights
    filters, biases = layer.get_weights()
    print("biases shzpe : ", biases.shape)
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


# https://www.arduino.cc/reference/en/language/functions/math/map/
def mapToRange(x, min_out, max_out):
    return (x - np.min(x)) * (max_out- min_out) / (np.max(x) - np.min(x)) + min_out

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def spec(N):                                             
    t = np.linspace(-510, 510, N)                                              
    return  tuple(map(tuple,np.clip(np.stack([-t, 510-np.abs(t), t], axis=1), 0, 1)))


print("starting")

filters = get_filter(LAYER)
print("Got filters")
fig = mlab.figure(1)
mlab.clf()


zdata = np.array([])
xdata = np.array([])
ydata = np.array([])

s_list = np.array([])

num_filters = filters.shape[-1]
num_channels = filters[:,:,:, 0].shape[-1]
c = np.linspace(0, 255, 256)


s = np.array([])
a = np.array([])
t = np.array([])

fig.scene.disable_render = True 
for i in range(num_filters):  # each filter in that channel 
    #print("i = ", i)

    filter_list = []
    sym_list = []
    anti_list  = []

    for j in range(CHANNEL, CHANNEL + 1): # Each channel (ex R G B)
    

        #print(j)
        f = filters[:,:,:, i]
        f = f[:,:, j]  

        sym, anti = getSymAntiSym(f)
        sym_mag = np.linalg.norm(sym) 
        anti_mag = np.linalg.norm(anti) 

        theta = getSobelAngle(f)
        theta = theta[theta.shape[0]//2, theta.shape[1]//2]

        # Data for three-dimensional scattered points
        zdata = np.append(zdata, sym_mag)
        ydata = np.append(ydata, anti_mag*np.sin(theta))
        xdata = np.append(xdata, anti_mag*np.cos(theta))


        filter_list.append(f)
        sym_list.append(sym)
        anti_list.append(anti)



###print(s_list[s_list>0.2])
#print(xdata[s_list>0.5], ydata[s_list>0.5], zdata[s_list>0.5])

glyphs = []
glyph_points = []

glyphs = mlab.points3d(xdata, ydata, zdata, color = (0.20,0.8,0.5), scale_factor=0.003)

#https://stackoverflow.com/questions/24308049/how-to-plot-proper-3d-axes-in-mayavi-like-those-found-in-matplotlib
lensoffset = 0
'''xx  = np.arange(np.min(xdata),np.max(xdata),0.01)
yy = np.arange(np.min(ydata),np.max(ydata),0.01)
zz = np.arange(np.min(zdata),np.max(zdata),0.01)

xy = xz = np.zeros_like(xx)    
yx = yz = np.zeros_like(yy)    
zx = zy = np.zeros_like(zz)    
mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.001,tube_radius=0.001, color=(1,0,0))
mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.001,tube_radius=0.001, color=(1,0,0))
mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.001,tube_radius=0.001, color=(1,0,0))'''

cursor3d = mlab.points3d(0., 0., 0., mode='axes',
                                color=(0, 0, 0),
                                scale_factor=0.5)


#glyph_points = glyphs.glyph.glyph_source.glyph_source.output.points.to_array()

print(xdata.shape,ydata.shape,zdata.shape )

#print(zdata.shape, xdata.shape, ydata.shape)
# Every object has been created, we can reenable the rendering.
#fig.scene.disable_render = False

# Here, we grab the points describing the individual glyph, to figure
# out how many points are in an individual glyph.


#points_dict[r] = (i, f, sym, anti)
#print(r.actor.actor._vtk_obj)



'''
Display Functions
'''
#https://docs.enthought.com/mayavi/mayavi/auto/example_select_red_balls.html
def picker_callback(picker_obj):

    fig, ax = plt.subplots(1,3)

    if picker_obj.actor in glyphs.actor.actors or picker_obj.actor in glyphs_red.actor.actors :

        # m.mlab_source.points is the points array underlying the vtk
        # dataset. GetPointId return the index in this array.
        point_id = picker_obj.point_id//num_filters
                                                            
        # If the no points have been selected, we have '-1'
        if point_id != -1:
            # Retrieve the coordinates coorresponding to that data
            # point
            x, y, z = xdata[point_id], ydata[point_id], zdata[point_id]
            #print(x,y,z, type(point_id), point_id, filter_list.shape)

            f = filter_list[point_id]
            sym = sym_list[point_id]
            anti_sym = anti_list[point_id]



            ax[0].imshow(f, cmap=plt.get_cmap('gray') )
            ax[0].set_title('Chanel : {}'.format(point_id))

            ax[1].imshow(sym, cmap=plt.get_cmap('gray') )
            ax[1].set_title('Chanel : {} ({:5.3f})'.format(point_id, np.linalg.norm(sym)))

            ax[2].imshow(anti_sym, cmap=plt.get_cmap('gray') )
            ax[2].set_title('Chanel : {} ({:5.3f})'.format(point_id, np.linalg.norm(anti_sym)))

            fig.show()

picker = fig.on_mouse_pick(picker_callback)

# Decrease the tolerance, so that we can more easily select a precise
# point.
picker.tolerance = 0.01

fig.scene.disable_render = False 
mlab.show()
