import numpy as np

def run():
    """Main program"""
    point = np.genfromtxt('housing.csv', delimiter=',')
    #hyperparameters
    learning_rate = 0.0001

if __name__ == '__main__':
    run()
