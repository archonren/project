__author__ = 'Guanhua, Joms'
from numpy import reshape
from struct import unpack
from numpy import *
import matplotlib.pyplot as plt
from scipy import ndimage

test_path_dat ="c:/data_road1/SLIC/um_000000.dat"
test_path_png = "c:/data_road1/SLIC/um_000000.png"
test_path_jpg = "c:/data_road1/SLIC/um_000000_SLIC.jpg"

png= ndimage.imread(test_path_png)
for x in ndindex((png.shape[0],png.shape[1])):
        png[x[0]][x[1]] = [0,0,130]


plt.imshow(png)
plt.show()

