import numpy as np
import cv2
import PhaseBased, utils

if __name__ ==  '__main__':
    '''
    This script magnifies the motion in `crane_crop.mp4` video 
    sequence.  This video, which is not uploaded, is available 
    for download at:
    http://people.csail.mit.edu/nwadhwa/phase-video/

    It takes around 10 minutes to complete on an average computer.
    '''
    input_path = r"C:\Users\schaller\Downloads\Source and Result Videos\Source and Result Videos\crane_crop.avi"
    output_path = 'crane_crop_magnified.mp4'
    monitor_path = 'monitor_04.mp4'
    
    alpha = 75
    D,N,K = 3,2,8
    fl,fh = 0.2, 0.25

    # We use a temporal filter of length greater than the video 
    # sequence to prevent bleeding, due to the narrow passband width.
    F_length = 1001
    
    frames, fs = utils.video2numpy(input_path)
    for i, frame in enumerate(frames):
        frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    frames = frames.astype(np.float32)

    F = PhaseBased.get_temporal_filter(fs,fh,fl,F_length)
    monitoring = None

    for c in range(frames.shape[3]):
        print('Processing channel %d' %(c))
        if c == 0:
            monitor = True
            frames[:, :, :, c], monitoring = PhaseBased.modify_motion(frames[:, :, :, c], alpha, D, N, K, F, monitor=monitor)
            mMin = monitoring.min()
            mMax = monitoring.max()
            monitoring = (monitoring - mMin)/(mMax-mMin)*255
            monitoring = np.moveaxis(np.broadcast_to(monitoring, (3, monitoring.shape[0], monitoring.shape[1], monitoring.shape[2])),0,-1).clip(0,255).astype(np.uint8)
            utils.numpy2video(monitor_path, monitoring, fs)
        else:
            monitor = False
            frames[:, :, :, c], _ = PhaseBased.modify_motion(frames[:, :, :, c], alpha, D, N, K, F, monitor=monitor)

    frames = frames.clip(0,255).astype(np.uint8)

    for i, frame in enumerate(frames):
        frames[i] = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
    utils.numpy2video(output_path,frames,fs)
#    if isinstance(monitoring, np.ndarray):
#        monitoring = np.broadcast_to(monitoring, (monitoring.shape[0], monitoring.shape[1], monitoring.shape[2], 3))
#        utils.numpy2video(monitor_path, monitoring, fs)
