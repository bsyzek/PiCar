from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import pygame
from drive_controller import DriveController

class PiCar:

    # Camera Info
    SCREEN_WIDTH  = 640
    SCREEN_HEIGHT = 480
    FRAME_RATE    = 32

    def __init__(self):
        print('Setting up PiCar...')

        self.dc = DriveController()
        self.setup_camera()

        # Using pygame in process remote keyboard inputs
        pygame.init()
        pygame.display.set_mode((100, 100))

    def setup_camera(self):
        print('Initializing camera...')

        self.camera = PiCamera()
        self.camera.resolution = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.camera.framerate = self.FRAME_RATE
        self.raw_capture = PiRGBArray(self.camera, size=(self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        # allow the camera to warm up
        time.sleep(1)

        print('Camera initialization complete...')
    
    def process_user_inputs(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                self.dc.stop()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.dc.forward()
                elif event.key == pygame.K_DOWN:
                    self.dc.backward()
                elif event.key == pygame.K_RIGHT:
                    self.dc.pivot_right()
                elif event.key == pygame.K_LEFT:
                    self.dc.pivot_left()


    def drive(self):
        for frame in self.camera.capture_continuous(self.raw_capture, format="bgr", use_video_port=True):

            # Display camera feed
            image = frame.array
            cv2.imshow("Feed", image)
            key = cv2.waitKey(1) & 0xFF
            self.raw_capture.truncate(0)

            # Exit program if user pressed 'q'
            if key == ord("q"):
                break

            # Use remote keyboard inputs to steer PiCar
            self.process_user_inputs() 
            

def main():
    car = PiCar()
    car.drive()

if __name__ == '__main__':
    main()
