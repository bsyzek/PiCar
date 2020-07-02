import argparse
from pi_car import PiCar

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--record_training_data", help="Boolean - Record training data from PiCar", action="store_true")
    parser.add_argument("-m", "--model", help="String - Name of model file")
    args = parser.parse_args()
    return args

def main(args):
   car = PiCar(record_training_data=args.record_training_data, model=args.model)
   car.drive()

if __name__ == '__main__':
    args = parse_args()
    main(args)
