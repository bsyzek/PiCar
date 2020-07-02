import tensorflow as tf

class ModelManager:

    def __init__(self):
        self.interpreter = None

    
    def load_model(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
       
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    
    def run_inference(self, image):
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output_data
    