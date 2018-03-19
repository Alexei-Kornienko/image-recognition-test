#!/usr/bin/env python

import sift
import homography
from PIL import Image
from numpy import *
from pylab import *

def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
    
    # if none of these cases they are equal, no filling needed.
    return concatenate((im1,im2), axis=1)

def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    """ Show a figure with lines joining the accepted matches
    input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
    matchscores (as output from 'match()'),
    show_below (if images should be shown below matches). """

    im3 = appendimages(im1,im2)
    if show_below:
        im3 = vstack((im3,im3))
    imshow(im3)
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot(
                [locs1[i][0],locs2[m[0]][0] + cols1],
                [locs1[i][1],locs2[m[0]][1]],
                'c'
            )
    axis('off')

imname = 'test-set/test1.png'
im1 = array(Image.open(imname).convert('L'))
sift.process_image(imname,'test1.sift')
l1,d1 = sift.read_features_from_file('test1.sift')

imname = 'samples/Screenshot from 2018-03-12 23-18-52.png'
im2 = array(Image.open(imname).convert('L'))
sift.process_image(imname,'test2.sift')
l2,d2 = sift.read_features_from_file('test2.sift')

matches = sift.match(d1, d2)

#figure()
#gray()
#plot_matches(im1, im2, l1, l2, matches, True)
#show()



