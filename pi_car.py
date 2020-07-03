import time
import cv2
import pygame
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from drive_controller import DriveController
from aws_manager import AWSManager
from model_manager import ModelManager

class PiCar:

    # Camera Info
    SCREEN_WIDTH  = 200
    SCREEN_HEIGHT = 66
    FRAME_RATE    = 24 

    def __init__(self, record_training_data=False, model=None):
        print('Setting up PiCar...')
        
        if record_training_data and not model:
            self.record_training_data = True
            print('Recording training data...')
        else:
            self.record_training_data = False

        if model:
            self.model_manager = ModelManager()
            self.model_manager.load_model(model)
            
        self.dc = DriveController()
        self.aws_manager = AWSManager()
        self.setup_camera()

        # Structs for collecting training images and labels
        self.training_images = []
        self.training_labels = []

        # Stores the current user drive instruction
        self.current_drive_input = 'forward'

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
                self.dc.forward()
                self.current_drive_input = 'forward'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.dc.pivot_right()
                    self.current_drive_input = 'right'
                elif event.key == pygame.K_LEFT:
                    self.dc.pivot_left()
                    self.current_drive_input = 'left'
                else:
                    self.dc.forward()
                    self.current_drive_input = 'forward'

    def convert_model_output_to_drive_command(self, model_output):
        print(str(model_output))

    def drive(self):
        for frame in self.camera.capture_continuous(self.raw_capture, format="bgr", use_video_port=True):

            image = frame.array

            if self.model_manager:
                # Use model to steer PiCar
                output = self.model_manager.run_inference(image)
                self.convert_model_output_to_drive_command(output)
            else: 
                # Use remote keyboard inputs to steer PiCar
                self.process_user_inputs() 

            # Record training data
            if self.record_training_data:
                self.training_images.append(image)
                self.training_labels.append(self.current_drive_input)
            
            # Display camera feed
            cv2.imshow("Feed", image)
            key = cv2.waitKey(1) & 0xFF
            self.raw_capture.truncate(0)

            # Exit program if user pressed 'q'
            if key == ord("q"):
                self.dc.stop()

                # Upload training data to AWS before exiting
                if self.record_training_data:
                    self.aws_manager.upload_training_data(self.training_images, self.training_labels)
                break
