#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import os
import video
from common import anorm2, draw_str
from time import clock
images_path="E:\image_0"
files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.5,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks_orb = []
        self.tracks_sift=[]
        self.tracks=[]
        #self.cam = video.create_capture(video_src)
        self.frame_idx = 0

    def run(self):
        i=0
        print(i)
        while i<(len(files)):
            #_ret, frame = self.cam.read()
            print(files[i])
            if i==173:
                print("false images")
                i=i+1
            else:
                frame=cv.imread(files[i])
                i=i+1
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                vis = frame.copy()

            if len(self.tracks_orb) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr1[-1] for tr1 in self.tracks_orb]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks_orb = []
                for tr1, (x, y), good_flag in zip(self.tracks_orb, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr1.append((x, y))
                    if len(tr1) > self.track_len:
                        del tr1[0]
                    new_tracks_orb.append(tr1)
                    cv.circle(vis, (x, y), 2, (0,255, 0), -1)
                self.tracks_orb = new_tracks_orb
                cv.polylines(vis, [np.int32(tr1) for tr1 in self.tracks_orb], False, (0, 255 , 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks_orb))
            
            if len(self.tracks_sift) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks_sift]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks_sift = []
                for tr, (x, y), good_flag in zip(self.tracks_sift, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks_sift.append(tr)
                    cv.circle(vis, (x, y), 2, (255, 0, 0), -1)
                self.tracks_sift = new_tracks_sift
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks_sift], False, (255,0 , 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks_sift))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                
                orb = cv.ORB_create()  # OpenCV 3 backward incompatibility: Do not create a detector with `cv2.ORB()`.
                kp = orb.detect(frame_gray, None)
                pts2= np.asarray([[p.pt[0], p.pt[1]] for p in kp])
                
                sift = cv.xfeatures2d.SIFT_create()
                kp = sift.detect(frame_gray, None)
                pts = np.asarray([[p.pt[0], p.pt[1]] for p in kp])

                final=np.array([])
                final=np.append(pts,pts2,axis=0)
                
                final2=np.around(final)

                final2=final2.astype(int)
                print(final2)
                final3=np.array([])
                
                for num in final2:
                    if num not in final3:
                        final3=np.append(final3,num,axis=0)
                        
                                            
#               print(kp)
#                print(des2)
                #img_building_keypoints = cv2.drawKeypoints(frame,key_points, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Draw circles.
                
                for x, y in [np.int32(tr[-1]) for tr in self.tracks_sift]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                #kp = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                
                for x, y in [np.int32(tr1[-1]) for tr1 in self.tracks_orb]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                if final is not None:
                    for x, y in np.float32(final).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                if pts is not None:
                    for x, y in np.float32(pts).reshape(-1, 2):
                        self.tracks_sift.append([(x, y)])

                if pts2 is not None:
                    for x, y in np.float32(pts2).reshape(-1, 2):
                        self.tracks_orb.append([(x, y)])
                


            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('try', mask)
            cv.imshow('lk_track', vis)

            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():

    App().run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    
    main()
    cv.destroyAllWindows()
