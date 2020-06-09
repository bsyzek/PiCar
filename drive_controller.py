import RPi.GPIO as gpio

class DriveController:

    LEFT_BACKWARD_PIN  = 7 
    LEFT_FORWARD_PIN   = 11
    RIGHT_FORWARD_PIN  = 13
    RIGHT_BACKWARD_PIN = 15 

    def __init__(self):
        self.setup_gpio()

    def __del__(self):
        gpio.cleanup()

    def setup_gpio(self):
        gpio.setmode(gpio.BOARD)

        gpio.setup(self.LEFT_FORWARD_PIN, gpio.OUT)
        gpio.setup(self.LEFT_BACKWARD_PIN, gpio.OUT)
        gpio.setup(self.RIGHT_FORWARD_PIN, gpio.OUT)
        gpio.setup(self.RIGHT_BACKWARD_PIN, gpio.OUT)

        self.left_forward_pwm = gpio.PWM(self.LEFT_FORWARD_PIN, 100) 
        self.left_backward_pwm = gpio.PWM(self.LEFT_BACKWARD_PIN, 100) 
        self.right_forward_pwm = gpio.PWM(self.RIGHT_FORWARD_PIN, 100) 
        self.right_backward_pwm = gpio.PWM(self.RIGHT_BACKWARD_PIN, 100) 

        self.left_forward_pwm.start(0)
        self.right_forward_pwm.start(0)
        self.left_backward_pwm.start(0)
        self.right_backward_pwm.start(0)

    def stop(self):
        self.left_forward_pwm.ChangeDutyCycle(0)
        self.right_forward_pwm.ChangeDutyCycle(0)
        self.left_backward_pwm.ChangeDutyCycle(0)
        self.right_backward_pwm.ChangeDutyCycle(0)

    def forward(self):
        self.left_forward_pwm.ChangeDutyCycle(100)
        self.right_forward_pwm.ChangeDutyCycle(100)
        self.left_backward_pwm.ChangeDutyCycle(0)
        self.right_backward_pwm.ChangeDutyCycle(0)

    def backward(self):
        self.left_forward_pwm.ChangeDutyCycle(0)
        self.right_forward_pwm.ChangeDutyCycle(0)
        self.left_backward_pwm.ChangeDutyCycle(100)
        self.right_backward_pwm.ChangeDutyCycle(100)

    def pivot_right(self):
        self.left_forward_pwm.ChangeDutyCycle(100)
        self.right_forward_pwm.ChangeDutyCycle(0)
        self.left_backward_pwm.ChangeDutyCycle(0)
        self.right_backward_pwm.ChangeDutyCycle(100)

    def pivot_left(self):
        self.left_forward_pwm.ChangeDutyCycle(0)
        self.right_forward_pwm.ChangeDutyCycle(100)
        self.left_backward_pwm.ChangeDutyCycle(100)
        self.right_backward_pwm.ChangeDutyCycle(0)
