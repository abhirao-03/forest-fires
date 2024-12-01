import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
import PIL
from perlin_noise.perlin_noise import PerlinNoise

# -----------------------------------------------------------------------------------------------------------
# JiaSun FUNCTIONS
# -----------------------------------------------------------------------------------------------------------

def generate_species(shape:tuple):
    noise = PerlinNoise(octaves=3)
    xpix, ypix = shape
    pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
    pic = np.array(pic)
    pic = np.abs(pic)
    species = np.zeros(shape = shape) 

    max_noise = np.max(pic)
    species[(pic<max_noise/2) & (pic>max_noise/6)] = 1
    species[(pic<max_noise/3*2) & (pic>max_noise/2)] = 2
    species[pic>max_noise/3*2] = 3

    return species

def generate_rain(shape:tuple):
    noise = PerlinNoise(octaves=2)
    xpix, ypix = shape
    pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
    pic = np.array(pic)
    pic = np.abs(pic)
    rain = np.zeros(shape = shape) 

    max_noise = np.max(pic)
    rain[(pic<max_noise/3*2) & (pic>max_noise/3)] = 1
    rain[(pic>max_noise/3*2)] = 2

    return rain

def start_fire_grid(grid,p=0.0001,burning_time=10):
    m = np.size(grid, 0)
    n = np.size(grid, 1)
    M = np.zeros([m, n])
    count_down = np.zeros([m, n])
    M_p = np.random.rand(m, n)
    M[M_p > (1 - p)] = 3
    M[grid > 0] = -100 #inflammable
    count_down[M == 3] = burning_time
    return M,count_down


def burning_prob(wind_dir): #[0,1] from south [1,0] from east
    '''
    prob = np.zeros(4)
    beta = 2

    p_x = 0.8*(1-np.exp(-beta*abs(wind_dir[0])))
    p_y = 0.8*(1-np.exp(-beta*abs(wind_dir[1])))
    prob[0] = random.uniform(0.5-np.sign(wind_dir[0])*p_x, 1) #x-
    prob[2] = random.uniform(0.5+np.sign(wind_dir[0])*p_x, 1) #x+
    prob[1] = random.uniform(0.5-np.sign(wind_dir[1])*p_y,1) #y-
    prob[3] = random.uniform(0.5+np.sign(wind_dir[1])*p_y,1) #y+
    return prob
    '''

    prob = np.zeros(6)
    distance = np.zeros(6)
    if wind_dir[0]!=0:
        theta = np.arctan(wind_dir[1]/wind_dir[0])
    else:
        if(wind_dir[1]>0):
            theta = np.pi/2
        else:
            theta = -np.pi/2

    if(wind_dir[0]<0):
        if(wind_dir[1]>0):
            theta = theta + np.pi
        else:
            theta = theta - np.pi
    
    theta = np.pi - theta

    distance[0] = abs(theta-2/3*np.pi)
    distance[1] = abs(theta-1/3*np.pi)
    distance[2] = abs(theta)
    distance[3] = abs(theta-5/3*np.pi)
    distance[4] = abs(theta-4/3*np.pi)
    distance[5] = abs(theta-np.pi)

    for i in range(6):
        if distance[i]>np.pi:
            distance[i] = 2*np.pi - distance[i]
        #prob[i] = random.uniform(1-distance[i]/np.pi,1)
        prob[i] = random.uniform(1-np.sin(distance[i]/2),1)
    
    return prob

def if_burning_around(M,i,j,wind_dir,tol = 0.9): #simplest
    tmp = 0
    m = np.size(M,0) #size
    n = np.size(M,1)
    prob = burning_prob(wind_dir)
    
    if i%2==0 :
        if i!=0 and M[i-1,j] == 3 and prob[0]>tol: #1
            tmp = 1
        elif i!=0 and j!=n-1 and M[i-1,j+1] == 3 and prob[1]>tol: #2
            tmp = 1
        elif j!=0 and M[i,j-1] == 3 and prob[5]>tol: #6
            tmp = 1
        elif j!=n-1 and M[i,j+1] == 3 and prob[2]>tol: #3
            tmp = 1  
        elif i!=m-1 and M[i+1,j] == 3 and prob[4]>tol: #5
            tmp = 1
        elif i!=m-1 and j!=n-1 and M[i+1,j+1] == 3 and prob[3]>tol: #4
            tmp = 1
    else:
        if i!=0 and j!=0 and M[i-1,j-1] == 3 and prob[0]>tol: #1
            tmp = 1
        elif i!=0 and M[i-1,j] == 3 and prob[1]>tol: #2
            tmp = 1
        elif j!=0 and M[i,j-1] == 3 and prob[5]>tol: #6
            tmp = 1
        elif j!=n-1 and M[i,j+1] == 3 and prob[2]>tol: #3
            tmp = 1
        elif i!=m-1 and j!=0 and M[i+1,j-1] == 3 and prob[4]>tol: #5
            tmp = 1
        elif i!=m-1 and M[i+1,j] == 3 and prob[3]>tol: #4
            tmp = 1

    if tmp==1 : return True
    else: return False

def update(M,count_down,species,rain,burning_time = 10,growing_time = 50,wind_dir=[0,0]):
    m = np.size(M,0) #size
    n = np.size(M,1)
    M_copy = np.zeros([m,n])
    M_copy[M==1/4] = 1/4
    M_copy[M==3/4] = 3/4
    M_copy[M==1/2] = 1/2
    M_copy[M==1] = 1
    M_copy[M==1 + 1/4] = 1 + 1/4
    M_copy[M==1 + 3/4] = 1 + 3/4
    M_copy[M==1 + 1/2] = 1 + 1/2
    M_copy[M==-1] = -1
    M_copy[M==4] = 4
    count_down_copy = count_down
    for i in np.arange(m):
        for j in np.arange(n):
            if(count_down_copy[i,j]!=0):
                count_down_copy[i,j] -= 1
            tol = 0.8+0.045*rain[i,j]
            if(M[i,j]>=0 and M[i,j]<1): #flammable
                if(if_burning_around(M,i,j,wind_dir,tol)):
                    M_copy[i,j] = 3
                    #count_down_copy[i,j] = burning_time*density[i,j]
                    count_down_copy[i,j] = burning_time[int(species[i,j])]
            elif(M[i,j]==3 and count_down_copy[i,j]==0):#burning
                #burnt time reached     
                M_copy[i,j] = 2 #burnt
            elif(M[i,j]==3 and count_down_copy[i,j]!=0):
                M_copy[i,j] = 3
            elif(M[i,j]==2): #burnt
                M_copy[i,j] = 1+species[i,j]/4 #growing
                #count_down_copy[i,j] = np.random.randint(growing_time-10,growing_time+10)
                if(species[i,j]!=0):
                    test = species[i,j]/4
                count_down_copy[i,j] = growing_time[int(species[i,j])]
            elif(M[i,j]>=1 and M[i,j]<2 and count_down_copy[i,j]==0): 
                M_copy[i,j] = species[i,j]/4 #flammable
            elif(M[i,j]>=1 and M[i,j]<2 and count_down_copy[i,j]!=0): 
                M_copy[i,j] = 1+species[i,j]/4
    return M_copy,count_down_copy

def plot_M(M,ax):
    ax.axis('off')
    '''
    m = np.size(M,0) #size
    n = np.size(M,1)
    line = []
    
    x,y = np.meshgrid(np.arange(m),np.arange(n))
    x = x.flatten()
    y = y.flatten()
    categories = M.flatten()

    color_map = {-1:'black',0:'y',1:'g',2:'b',3:'black'}
    colors = [color_map[cat] for cat in categories]
    for cat in color_map:
        mask = categories == cat
        line.append(ax.scatter(x[mask], y[mask], c=color_map[cat], label=cat,s=0.1))
    '''
    
    return ax.imshow(M,cmap='Reds')

def plot_mesh(M):
    m = np.size(M,0) #size
    n = np.size(M,1)

    x,y = np.meshgrid(np.arange(m),np.arange(n))

    plt.pcolormesh(x, y, M)
    plt.show()

def wind(i,N):
    theta = np.pi/N*i
    return [50*np.cos(theta),50*np.sin(theta)]

# -----------------------------------------------------------------------------------------------------------
# Willow FUNCTIONS
# -----------------------------------------------------------------------------------------------------------


def get_cell(board, row, col):
    '''
    Return the value of the cell at (row, col) in board, with non-periodic boundaries.
    If (row, col) is out of bounds, return 0 (considered as non-flammable).
    '''
    if row < 0 or row >= board.shape[0] or col < 0 or col >= board.shape[1]:
        return 100 # Out of bounds cells are treated as non-flammable
    return board[row, col]

def set_cell(board, row, col, value):
    '''
    Assign the given value to the cell at (row, col) in board, with non-periodic boundaries.
    If (row, col) is out of bounds, do nothing.
    '''
    if 0 <= row < board.shape[0] and 0 <= col < board.shape[1]:
        board[row, col] = value
    return board


# -----------------------------------------------------------------------------------------------------------
# Facets of Mathematics FUNCTIONS
# -----------------------------------------------------------------------------------------------------------
 
def _img_a_cast(img_a, dtype, true_color=False):
    """
    This function checks and corrects for any errors in our image format.
    """

    # Make sure the max and min are 255 and 0 for any given value.
    img_a = np.maximum(img_a, 0)                        
    img_a = np.minimum(img_a, 255)


    # round any floats to integers
    img_a = np.round(img_a, 0)
    img_a = np.array(img_a + 1.0e-6, dtype=dtype)

    # handles grayscale images
    if len(img_a.shape) == 2:
        if true_color:
            img_a_gs = np.zeros((img_a.shape[0], img_a.shape[1], 3),
                                dtype=dtype)
            for k in range(3):
                img_a_gs[:, :, k] = img_a
            return img_a_gs
        else:
            return img_a
        
    # tests for unexpected image types by seeing if there's more than RGB channels
    # or if the last shape has more than 3 channels.
    else:
        if len(img_a.shape) != 3 or img_a.shape[2] != 3:
            raise RuntimeError("Unexpected image type")
        return img_a


def linear_filter(img_a, W, **kwargs):
    """
    Applies a convolution with a kernel W to the supplied image
    """

    img_a = _img_a_cast(img_a, dtype=np.int64)
    W = np.fliplr(np.flipud(W))

    if len(img_a.shape) == 2:
        img_filtered_a = scipy.ndimage.convolve(img_a, W, **kwargs)
    else:
        if len(img_a.shape) != 3 or img_a.shape[2] != 3:
            raise RuntimeError("Unexpected image type")
        img_filtered_a = np.zeros_like(img_a)
        for k in range(3):
            img_filtered_a[:, :, k] = scipy.ndimage.convolve(
                img_a[:, :, k], W, **kwargs)

    return _img_a_cast(img_filtered_a, dtype=np.int64)

def imwrite(fp, img_a, **kwargs):
    img_a = _img_a_cast(img_a, dtype=np.uint8)
    img = PIL.Image.fromarray(img_a)
    img.save(fp, **kwargs)


# -----------------------------------------------------------------------------------------------------------
# ALTERNATE SOURCE FUNCTIONS
# -----------------------------------------------------------------------------------------------------------

def bin_array_to_shape(arr, target_shape):
    """
    https://stackoverflow.com/questions/36063658/how-to-bin-a-2d-array-in-numpy
    https://scipython.com/blog/binning-a-2d-array-in-numpy/
    """
    target_rows, target_cols = target_shape
    input_rows, input_cols = arr.shape

    # Compute binning factors
    row_factor = input_rows // target_rows
    col_factor = input_cols // target_cols

    # Bin the array using reshaping and mean
    binned = arr[:row_factor * target_rows, :col_factor * target_cols]  # Crop to exact multiples
    binned = binned.reshape(target_rows, row_factor, target_cols, col_factor).mean(axis=(1, 3))

    return binned