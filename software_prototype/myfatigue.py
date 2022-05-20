from scipy.spatial import distance as dist
import numpy as np
import argparse
from collections import OrderedDict
import dlib
import cv2
import math
from threading import Thread
# dlib.DLIB_USE_CUDA = False

# Parameter for blinking
EYE_AR_THRESH = 0.2  # Threshold for eye area ratio
EYE_AR_CONSEC_FRAMES = 2  # Frame Count Threshold

# Parameter for yawning
MAR_AR_THRESH = 0.55  # Threshold for mouth area ratio
MOUTH_AR_CONSEC_FRAMES = 20  # Frame Count Threshold

# Parameter for counter
ROLL_COUNTER = 0  # Couter for blinking frames
TOTAL = 0  # Counter for blinking times
ROLL_mCOUNTER = 0  # Counter for yawning frames
mTOTAL = 0  # Counter for yawning times

NO_EYE_COUNTER = 0

# label the point
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def eye_aspect_ratio(eye):
    # horizontal distance(Euclidean coordinates)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # vertical distance(Euclidean coordinates)
    C = dist.euclidean(eye[0], eye[3])
    # ear: eye length to width ratio calculation
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar


# input parameters
# face_landmark_path = './shape_predictor_68_face_landmarks.dat'
# video_path = './4.mp4'
# face_landmark_path = r'D:\Wangysh\Semester8\project\test\shape_predictor_68_face_landmarks.dat'
# video_path = r'D:\Wangysh\Semester8\project\test\4.mp4'

# Initialise dlib's face detector (HOG), then create facial marker predictions
# print("[INFO] loading facial landmark predictor...")
# Use dlib.get_frontal_face_detector() to get the face position detector
detector = dlib.get_frontal_face_detector()
# Obtaining a face feature position detector using dlib.shape_predictor
predictor = dlib.shape_predictor(r'models/shape_predictor_68_face_landmarks.dat')
# Separate indexes for left and right eye facial markings
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
(nStart, nEnd) = FACIAL_LANDMARKS_68_IDXS["nose"]
(mStart, mEnd) = FACIAL_LANDMARKS_68_IDXS["mouth"]
(rbStart, rbEnd) = FACIAL_LANDMARKS_68_IDXS["right_eyebrow"]
(lbStart, lbEnd) = FACIAL_LANDMARKS_68_IDXS["left_eyebrow"]
(jStart, jEnd) = FACIAL_LANDMARKS_68_IDXS["jaw"]


# Converts shape to coordinate point form(array)
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # get the coordinate
    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


# Loop frames from the video stream
def detfatigue(frame):
    # get global parameters
    global EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, MAR_AR_THRESH, MOUTH_AR_CONSEC_FRAMES, ROLL_COUNTER, TOTAL, ROLL_mCOUNTER, mTOTAL, Rolleye, Rollmouth, NO_EYE

    # frame = imutils.resize(frame, width=720)
    # Converting BGR format to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect face
    rects = detector(gray, 0)
    COUNTER = 0
    mCOUNTER = 0
    if len(rects) == 0:
        NO_EYE_COUNTER = 1
    else:
        NO_EYE_COUNTER = 0

    eyear = 0.0
    mouthar = 0.0
    # Loop through the face position information and 
    # use predictor(gray, rect) to obtain information about the position of the face features
    for rect in rects:
        # get the cooridinator
        shape = predictor(gray, rect)
        # Converting facial features into an array format
        shape = shape_to_np(shape)

        # cooridinator for each feature
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        Nose = shape[nStart:nEnd]
        Mouth = shape[mStart:mEnd]
        rightEyebrow = shape[rbStart:rbEnd]
        leftEyebrow = shape[lbStart:lbEnd]
        Jaw = shape[jStart:jEnd]

        # The constructor calculates the EAR values for the left and right eyes, 
        # using the average value as the final EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        eyear = (leftEAR + rightEAR) / 2.0
        # print(eyear)

        # Mouth EAR
        mouthar = mouth_aspect_ratio(Mouth)

        # Fetigue detection
        # blinking detection
        if eyear < EYE_AR_THRESH:
            # If the eye opening and closing is less than threshold
            # Then COUNTER + 1 means in this frame, the eyes is closed
            ROLL_COUNTER += 1
            if ROLL_COUNTER < EYE_AR_CONSEC_FRAMES:
                COUNTER = 0
            elif ROLL_COUNTER == EYE_AR_CONSEC_FRAMES:
                COUNTER = EYE_AR_CONSEC_FRAMES
            else:
                COUNTER = 1

        else:
            # If 2 consecutive times are less than the threshold, 
            # then a blink has been performed
            if ROLL_COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                # Resetting the eye frame counter
                ROLL_COUNTER = 0

        # Mouth detector(yawning detector) 
        if mouthar > MAR_AR_THRESH:
            ROLL_mCOUNTER += 1
            if ROLL_mCOUNTER < MOUTH_AR_CONSEC_FRAMES:
                mCOUNTER = 0
            elif ROLL_mCOUNTER == MOUTH_AR_CONSEC_FRAMES:
                mCOUNTER = MOUTH_AR_CONSEC_FRAMES
            else:
                mCOUNTER = 1
        else:
            # If n consecutive times are less than the threshold, means a yawning action
            if ROLL_mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                mTOTAL += 1
                ROLL_mCOUNTER = 0

        # Show in cv2
        # Use cv2.convexHull to get the position of the convex pack and 
        # drawContours to draw the outline position for the drawing operation
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        noseHull = cv2.convexHull(Nose)
        mouthHull = cv2.convexHull(Mouth)
        rightEyebrowHull = cv2.convexHull(rightEyebrow)
        leftEyebrowHull = cv2.convexHull(leftEyebrow)
        jawHull = cv2.convexHull(Jaw)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyebrowHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [leftEyebrowHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)

        cv2.line(frame, tuple(shape[38]), tuple(shape[40]), (0, 255, 0), 1)
        cv2.line(frame, tuple(shape[43]), tuple(shape[47]), (0, 255, 0), 1)
        cv2.line(frame, tuple(shape[51]), tuple(shape[57]), (0, 255, 0), 1)
        cv2.line(frame, tuple(shape[48]), tuple(shape[54]), (0, 255, 0), 1)

    # frame is the video with face and detector
    return (frame, TOTAL, mTOTAL, COUNTER, mCOUNTER, NO_EYE_COUNTER)
