#import cv2



# Install mayavi and vtk from https://www.lfd.uci.edu/~gohlke/pythonlibs/#vtk
# pip install "wheels package"

from msilib.schema import Error
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

LAYER = 1


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

#https://stackoverflow.com/questions/24308049/how-to-plot-proper-3d-axes-in-mayavi-like-those-found-in-matplotlib
lensoffset = 0
xx = yy = zz = np.arange(-1.5,1.5,0.1)
xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=0.01)
mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=0.01)
mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=0.01)

cursor3d = mlab.points3d(0., 0., 0., mode='axes',
                                color=(0, 0, 0),
                                scale_factor=0.5)


filter_list = dict()
sym_list = dict()
anti_list = dict()

zdata = np.array([])
xdata = np.array([])
ydata = np.array([])

glyphs = dict()


num_filters = filters.shape[-1]
num_channels = filters[:,:,:, 0].shape[-1]
c = np.linspace(0, 255, 256)
for i in range(num_channels):  # Each channel (ex R G B)
    print("i = ", i)
    z = np.array([])
    x = np.array([])
    y = np.array([])

    filter_list[i] = []
    sym_list[i] = []
    anti_list[i] = []


    for j in range(num_filters): #each filter in that channel

        print(j)
        f = filters[:,:,:, j]
        f = f[:,:, i]  

        sym, anti = getSymAntiSym(f) # getSymAntiSym(normalizeToOne(f))
        sym_mag = np.linalg.norm(sym) 
        anti_mag = np.linalg.norm(anti) 

        theta = getSobelAngle(f)
        theta = theta[theta.shape[0]//2, theta.shape[1]//2]

        # Data for three-dimensional scattered points
        z = np.append(z, sym_mag)
        y = np.append(y, anti_mag*np.sin(theta))
        x = np.append(x, anti_mag*np.cos(theta) )

        filter_list[i].append(f)
        sym_list[i].append(sym)
        anti_list[i].append(anti)
    
    glyphs[i] = mlab.points3d(x, y, z,  color = spec(num_channels)[i], scale_factor=.1)

    zdata = np.append(zdata, z)
    ydata = np.append(ydata, y)
    xdata = np.append(xdata, x)

    glyph_points = glyphs[i].glyph.glyph_source.glyph_source.output.points.to_array()

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
    fig, ax = plt.subplots(ncols=3)

    for n, g in glyphs.items():
        if picker_obj.actor in g.actor.actors :

            # m.mlab_source.points is the points array underlying the vtk
            # dataset. GetPointId return the index in this array.
            point_id = picker_obj.point_id//glyph_points.shape[0]
                                                                
            # If the no points have been selected, we have '-1'
            if point_id != -1:
                # Retrieve the coordinates coorresponding to that data
                # point
                x, y, z = xdata[point_id], ydata[point_id], zdata[point_id]
                #print(x,y,z, type(point_id), point_id, filter_list.shape)

                f = filter_list.get(n)[point_id]
                sym = sym_list.get(n)[point_id]
                anti_sym = anti_list.get(n)[point_id]

                ax[0].imshow(f, cmap=plt.get_cmap('gray'))
                if LAYER == 1: 
                    ax[0].set_title('Filter : {} , Chanel : {}'.format(point_id, RGB[n]))
                else:
                    ax[0].set_title('Filter : {} , Chanel : {}'.format(point_id, n))
                ax[1].imshow(sym, cmap=plt.get_cmap('gray')) 
                ax[2].imshow(anti_sym, cmap=plt.get_cmap('gray')) 
                fig.show()


picker = fig.on_mouse_pick(picker_callback)

# Decrease the tolerance, so that we can more easily select a precise
# point.
picker.tolerance = 0.01

mlab.show()
