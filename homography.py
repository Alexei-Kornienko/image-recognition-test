#!/usr/bin/env python

import ransac
from PIL import Image
from numpy import *
from pylab import *


class RansacModel(object):
    """ Class for testing homography fit with ransac.py from
    http://www.scipy.org/Cookbook/RANSAC"""
    def __init__(self,debug=False):
        self.debug = debug

    def fit(self, data):
        """ Fit homography to four selected correspondences. """
        # transpose to fit H_from_points()
        data = data.T
        # from points
        fp = data[:3,:4]
        # target points
        tp = data[3:,:4]
        # fit homography and return
        return H_from_points(fp,tp)

    def get_error( self, data, H):
        """ Apply homography to all correspondences,
        return error for each transformed point. """
        data = data.T
        # from points
        fp = data[:3]
        # target points
        tp = data[3:]
        # transform fp
        fp_transformed = dot(H,fp)
        # normalize hom. coordinates
        for i in range(3):
            fp_transformed[i] /= fp_transformed[2]
        # return error per point
        return sqrt( sum((tp-fp_transformed)**2,axis=0) )


def H_from_ransac(fp,tp,model,maxiter=1000,match_theshold=10):
    """ Robust estimation of homography H from point
    correspondences using RANSAC (ransac.py from
    http://www.scipy.org/Cookbook/RANSAC).
    input: fp,tp (3*n arrays) points in hom. coordinates. """
    # group corresponding points
    data = vstack((fp,tp))
    # compute H and return
    H,ransac_data = ransac.ransac(data.T,model,4,maxiter,match_theshold,10,return_all=True)
    return H,ransac_data['inliers']


def normalize(points):
    """ Normalize a collection of points in
    homogeneous coordinates so that last row = 1. """
    for row in points:
        row /= points[-1]
    return points

def make_homog(points):
    """ Convert a set of points (dim*n array) to
    homogeneous coordinates. """
    return vstack((points,ones((1,points.shape[1]))))


def H_from_points(fp,tp):
    """ Find homography H, such that fp is mapped to tp
    using the linear DLT method. Points are conditioned
    automatically. """
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
    # condition points (important for numerical reasons)
    # --from points--
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = dot(C1,fp)
    # --to points--
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9

    C2 = diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = dot(C2,tp)
    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = zeros((2*nbr_correspondences,9))
    for i in range(nbr_correspondences):
        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,
                tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,
                tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]

    U,S,V = linalg.svd(A)
    H = V[8].reshape((3,3))
    # decondition
    H = dot(linalg.inv(C2),dot(H,C1))
    # normalize and return
    return H / H[2,2]
