import numpy as np
import cv2 as cv
import threading
import time
import sys

def calculate_optical_flow(frame, winsize, thread_name):
    prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    
    start_time = time.time()

    while True:
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # Calculates the dense optical flow using the Gunnar Farneback’s algorithm
        flow = cv.calcOpticalFlowFarneback(
            prev=prvs, 
            next=next, 
            flow=None, 
            pyr_scale=0.5, 
            levels=8, 
            winsize=winsize,  # change between 8 and 32
            iterations=3, 
            poly_n=5,
            poly_sigma=1.2, 
            flags=0)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        
        # Show image and handle key events in the main thread
        cv.imshow(f'Optical Flow - WinSize {winsize} - {thread_name}', bgr)
        
        k = cv.waitKey(30) & 0xff
        
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite(f'optical_flow_{winsize}_{thread_name}.png', bgr)
        
        prvs = next

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'{thread_name} elapsed time: {elapsed_time} seconds')


def check_elapsed_time():
    start_time = time.time()
    while True:
        time.sleep(1)
        elapsed_time = time.time() - start_time
        if elapsed_time > 30:
            print("Terminating threads due to timeout.")
            cap.release()
            cv.destroyAllWindows()
            sys.exit()

# Open video capture
cap = cv.VideoCapture(1)

# Read the first frame
ret, frame1 = cap.read()

# Create two threads for each window size and the time-checking thread
thread1 = threading.Thread(target=calculate_optical_flow, args=(frame1, 8, 'Thread 1'))
thread2 = threading.Thread(target=calculate_optical_flow, args=(frame1, 32, 'Thread 2'))
time_thread = threading.Thread(target=check_elapsed_time)

# Start both threads
thread1.start()
thread2.start()
time_thread.start()

# Wait for both threads to finish
thread1.join()
thread2.join()
time_thread.join()
