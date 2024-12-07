## Script to process stomp video without processing large motions and without using all the memory
#
# Neal Wadhwa, April 2013

import textwrap
import argparse
from Video import Video
from pathlib import Path
from scipy import signal
from skimage import color
import numpy as np

parser = argparse.ArgumentParser(description=textwrap.dedent('''\
    Phase Amplify Large Motions.
'''))

#vidFile = inFile
vidName = 'stomp'
#outDir = resultsDir
sigma = 12 # Blurring
alpha = 50 # Magnification
FS = 300 # Sampling rate
fl = 1 # Freq bands
fh = 25 # Freq bands

def getFilterIDX( filters ):

    nFilts = math.max(len(filters))
    filtIDX = np.cell(nFilts, 2)
    croppedFilters = np.cell(nFilts,1)

    for k in range(1,nFilts):
        indices = getIDXFromFilter(filters[k])
        filtIDX[k,1] = indices[1]
        filtIDX[k,2] = indices[2]
        croppedFilters[k] = filters[k](indices[1], indices[2])

def  getIDXFromFilter(filter ):
    aboveZero = filter>1e-10
    dim1 = sum(aboveZero,2)>0
    dim1 = dim1 or np.rot90(dim1,2)
    dim2 = sum(aboveZero,1)>0
    dim2 = dim2 or np.rot90(dim2,2)
    dims = filter.shape
    filtIDX = []
    idx1 = range(1, dims[0])
    idx2 = range(1, dims[1])

    idx1 = idx1(dim1);
    idx1 = range(math.min(idx1),math.max(idx1))

    idx2 = idx2(dim2);
    idx2 = range(math.min(idx2),math.max(idx2))

    filtIDX[0] = idx1;
    filtIDX[1] = idx2;

    return filtIDX

def clip(im, minVal, maxVal):

    if maxVal < minVal:
        raise Exception('MAXVAL should be less than MINVAL')
    res = im;
    res[im < minVal] = minVal;
    res[im > maxVal] = maxVal;

    return res

def getPolarGrid(dimension):
    center = np.ceil((dimension + 0.5) / 2);

    # Create rectangular grid
    [xramp, yramp] = np.meshgrid((np.array(range(1,dimension[1])-center[1]))/ (dimension[2] / 2), (np.array(range(1, dimension[0]))-center[0])/ (dimension[0] / 2) )
    # Convert to polar coordinates
    angle = np.arctan2(yramp, xramp)
    rad = np.sqrt(xramp**2 + yramp**2)
    # Eliminate places where rad is zero, so logarithm is well defined
    rad[center[0], center[1]] = rad[center[0], center[1] - 1]

    return angle, rad


def getRadialMaskPair( r, rad, twidth):
    log_rad = np.log2(rad) - np.log2(r);

    himask = log_rad;
    himask = clip(himask, -twidth, 0);
    himask = himask * np.pi / (2 * twidth);
    himask = abs(np.cos(himask));
    lomask = np.sqrt(1 - himask**2);

    return himask, lomask


def getAngleMask(b,  orientations, angle):

    order = orientations-1;
    const = (2**(2*order))*(math.factorial(order)**2)/(orientations*math.factorial(2*order)); # Scaling constant
    angle = (np.pi+angle - np.pi*(b-1)/orientations) % 2*np.pi - np.pi; # Mask angle mask
    anglemask = 2*np.sqrt(const)*np.cos(angle)**order*(abs(angle)<np.pi/2);  # Make falloff smooth
    return anglemask

def getFiltersdimension (rVals, orientations, kvargs):
    filters = []
    #p = inputParser;

    defaultTwidth = 1; #Controls falloff of filters

    # addRequired(p, 'dimension')
    # addRequired(p, 'rVals')
    # addRequired(p, 'order')
    # addOptional(p, 'twidth', defaultTwidth, @isnumeric)
    # parse(p, dimension, rVals, orientations, varargin{:})

    dimension = kvargs['dimension']
    rVals = kvargs['rVals']
    orientations = kvargs['order']
    twidth = p.Results.twidth


    [angle, log_rad] = getPolarGrid(dimension) # Get polar coordinates of frequency plane
    count = 1
    [himask, lomaskPrev] = getRadialMaskPair(rVals[1], log_rad, twidth)
    filters[count] = himask
    count += 1
    for k in range(2, max(len(rVals))):
        [himask, lomask] = getRadialMaskPair(rVals[k], log_rad, twidth)
        radMask = himask*lomaskPrev
        for j in range(1,orientations):
            anglemask = getAngleMask(j, orientations, angle)
            filters[count] = radMask*anglemask/2
            count += 1
        lomaskPrev = lomask

    filters[count] = lomask
    return filters


def  buildSCFpyrGen(im, croppedFilters, filtIDX, kwargs):

    nFilts = math.max(len(croppedFilters))

    # Parse optional arguments
    # p = inputParser
    defaultInputIsFreqDomain = False

    # addOptional(p, 'inputFreqDomain', defaultInputIsFreqDomain, @ islogical)

    # parse(p, varargin{:})
    isFreqDomain = defaultInputIsFreqDomain # p.Results.inputFreqDomain

    # Return pyramid in the usual format of a stack of column vectors

    if (isFreqDomain):
        imdft = im
    else:
        imdft = np.fftshift(np.fft2(im)) # DFT of image end

    pyr = []
    pind = np.zeros(nFilts, 2)
    for k in range(1,nFilts):
        tempDFT = croppedFilters[k]*imdft(filtIDX[k,0], filtIDX[k,1])   # Transform domain
        curResult = np.ifft2(np.ifftshift(tempDFT))
        # pyr{k} = curResult[:]
        # pind{k} = size(curResult)
        pind[k,:] = len(curResult)
        pyr.append(curResult[:])

def pyrBandIndices(pind,band)

    if (band > pind.shape[0]) or (band < 1):
        raise Exception(f"BAND_NUM must be between 1 and number of pyramid bands {pind.shape[0]}.")
    if pind.shape[1] != 2:
        raise Exception(f"INDICES must be an Nx2 matrix indicating the size of the pyramid subbands")
    ind = 1
    for l in range(1,band-1):
        ind = ind + np.prod(pind[l,:], axis=0)
    indices = range(ind,ind+np.prod(pind[band,:], axis=0)-1)
    return indices

def pyrBand(pyr, pind, band):
    return np.reshape( pyr(pyrBandIndices(pind,band)), pind(band,1), pind(band,2) )


def AmplitudeWeightedBlur( inA, weight, sigma )
# AMPLITUDEWEIGHTEDBLUR Summary of this function goes here
#   Detailed explanation goes here

    if sigma!=0:
        kernel = fspecial('gaussian', np.ceil(4*sigma), sigma)
        sz = kernel.shape
        weight = weight+eps
        out = imfilter(inA*weight, kernel,'circular')
        weightMat = imfilter(weight,kernel,'circular')
        out = out/weightMat
    else:
        out = inA
    return out

if __name__ == '__main__':
    args = parser.parse_args()
    vr = Video(Path(args.in_filename).resolve())
    #vr = VideoReader(vidFile)

    vid = vr.videoBuffer()
    [h, w, nC, nF] = vid.shape
    toD = lambda k :  color.rgb2gray(vid[:,:,:, k]) # Get luma component of a frame
    getChroma = lambda k : color.rgb2yuv((vid[:,:,:, k]))

    phases = None
    amps = None

    filters = getFilters((h, w), 2**range(0,-3, -1), 4)
    [croppedFilters, filtIDX] = getFilterIDX(filters)
    buildPyr = lambda im : buildSCFpyrGen(im, croppedFilters, filtIDX)
    reconPyr = lambda pyr, pind : reconSCFpyrGen(pyr, pind, croppedFilters, filtIDX)
    refFrame = 200

    tag = f"alpha{alpha}-sigma{sigma}-band{fl}-{fh}-refFrame{refFrame}"

    #mkdir(fullfile(outDir, tag))

    # vw_withLarge = VideoWriter(fullfile(outDir, sprintf('%s-%s-withlarge.avi', vidName, tag)))
    #vw_withLarge.Quality = 90
    #vw_withLarge.FrameRate = 30
    #vw_withLarge.open()
    #vw_withLarge.writeVideo(vid(:,:,:, 1: 2))


    #vw_withoutLarge = VideoWriter(fullfile(outDir, sprintf('%s-%s-withoutlarge.avi', vidName, tag)))
    #vw_withoutLarge.Quality = 90
    #vw_withoutLarge.FrameRate = 30
    #vw_withoutLarge.open()
    #vw_withoutLarge.writeVideo(vid(:,:,:, 1: 2))

    [B, A] = signal.butter(1, [fl / FS * 2, fh / FS * 2]) # Temporal Filter

    refPyr = buildPyr(toD(refFrame))
    pyr = buildPyr(toD(1))
    phaseM2 = np.angle(pyr/refPyr)
    [pyr, pind] = buildPyr(toD(2))
    phaseM1 = np.angle(pyr/ refPyr)

    # Initialize butterworth filter
    outPhaseM2 = phaseM2
    outPhaseM1 = phaseM1
    outPhaseM3 = phaseM1
    outPhaseM4 = phaseM1

    # Amplification
    for k in range(3,nF):
        print(f"Processing frame {k}\n")
        curPyr = buildPyr(toD(k))
        curPhase = np.angle(curPyr/refPyr)
        # Butterworth filter temporally
        outPhase = (B[1] * curPhase + B[2] * phaseM1 + B[3] * phaseM2 - A[2] * outPhaseM1 - A[3] * outPhaseM2) / A[1]
        phaseM2 = phaseM1
        phaseM1 = curPhase
        outPhaseM5 = outPhaseM4
        outPhaseM4 = outPhaseM3
        outPhaseM3 = outPhaseM2
        outPhaseM2 = outPhaseM1
        outPhaseM1 = outPhase

        # Spatial Blurring
        for band in range(2, len(pind))
            idx = pyrBandIndices(pind, band)
            temp = pyrBand(outPhase, pind, band)
            curAmp = pyrBand(abs(curPyr), pind, band)
            temp = AmplitudeWeightedBlur(temp, curAmp, sigma)
            outPhase[idx] = temp        # temp[:]

        # Reconstruction with processing large motions
        outPhase = outPhase * (alpha)
        luma = (reconPyr(curPyr. * exp(1i * outPhase), pind))
        frame = getChroma(k)
        frame(:,:, 1) = luma
        # vw_withLarge.writeVideo(im2uint8(ntsc2rgb(frame)))

        # Reconstruction without processing large motions
        # Spatiotemporally smooth phases to increase robustness
        phaseVar = (abs(outPhase) + abs(outPhaseM2) + abs(outPhaseM3) + abs(outPhaseM4) + abs(outPhaseM5)) / 5

        for band = 2:size(pind, 1)
            idx = pyrBandIndices(pind, band)
            temp = pyrBand(phaseVar, pind, band)
            curAmp = pyrBand(abs(curPyr), pind, band)
            temp = AmplitudeWeightedBlur(temp, curAmp, sigma)
            phaseVar(idx) = temp(:)


        cutoff = pi
        for band = 1:3
            for or = 1:4
                idx = pyrBandIndices(pind, 1 + or + 4 * (band - 1))
                temp = outPhase(idx)
                temp(phaseVar(idx) > cutoff / 2. ^ band) = 0
                outPhase(idx) = temp


        luma = (reconPyr(curPyr. * exp(1i * outPhase), pind))
        frame = getChroma(k)
        frame(:,:, 1) = luma
        # vw_withoutLarge.writeVideo(im2uint8(ntsc2rgb(frame)))


    # vw_withLarge.close()
    # vw_withoutLarge.close()