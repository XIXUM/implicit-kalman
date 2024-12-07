"""Implicit Kalman Class

        (c) felixschaller.com 2023
        resolves 3D environment pixelwise by extracting optical flow from a sequence of images

        authors: Felix Schaller, Yusuf Can Simsek
"""

import numpy as np
from skimage.registration import phase_cross_correlation, optical_flow_tvl1

class OpticalFlow:
    """
    Optical Flow class derives a vector field from optical flow
    """
    def __init__(self):
        pass

    def flowResolve(self, imageA: nparray, imageB: nparray):

        # pixel precision, subpixel also available
        # shift, error, diffphase = phase_cross_correlation(imageA, imageB)

        v, u = optical_flow_tvl1(a, b)

        return (v, u)



class ImplicitKalman:

    # 3D Point Cloud with RGB Voxels
    pointCloud = None

    # 3D Spline Curve
    cameraTrack = None

    #Derived field of view, brute force over time in sequence
    FOV = None

    # Store PointCloud in an OcTree Environment for faster access and 3D Navigation
    OcTreeEnv = None

    # Class for flow resolve
    flowResolver = None

    v,u = None

    def __init__(self):
        flowResolver = OpticalFlow()

    def cameraResolve(self, imageSequence):
        """
        resolves a 3D environment and a camera track from a sequence of Images and stores the information in the class
        :param imageSequence:
        :return:
        """
        for i in range(0, imageSequence.shape[0]-1):
            imageA = imageSequence[i]
            imageB = imageSequence[i+1]

            self.v, self.u = self.flowResolver.flowResolve(imageA, imageB)

            zV, zU = self._calculateDepth()



    def getPointCloud(self):
        """
        returns a RGB voxel point cloud
        :return pointCloud:
        """

    def getCameraTrack(self):
        """
        returns spline curve of the Camera Track
        :return:
        """

    def getCameraInfo(self):
        """
        returns the Camera Info as dictionary
        :return dict: {
                        'FOV' : field of view
                        'AOV' : angle of view
                        'width': backplate height in pixels (or maybe mm)
                        'heigt': backplate heigt in pixels (or maybe mm)
                        ...more infos
                        }:
        """

    def _calculateDepth(self):
        pass