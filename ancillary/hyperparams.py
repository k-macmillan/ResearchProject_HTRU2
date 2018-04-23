import random
import sys
from numpy import random as np

class HyperParams: 
    def __init__(self, seed=None):
        """Fills all self variables in self.params. Sets seed member so it can be appended to workerData folder"""        
        random.seed(seed)        
        self.params = { 'learning_rate_min'     : 0.01,
                        'learning_rate_max'     : 0.1,
                        'h_nodes_layer_min'     : 12,
                        'h_nodes_layer_max'     : 36,
                        'base_explr_min'        : 0.30,
                        'base_explr_max'        : 0.99,
                        'rand_explr_A_min'      : 1.0,
                        'rand_explr_A_max'      : 4.0,
                        'rand_explr_B_min'      : 4.0,
                        'rand_explr_B_max'      : 10.0,
                        'acc_coef_min'          : 1.0,
                        'acc_coef_max'          : 1.0,
                        'advantage_coef_min'    : 2.0,
                        'advantage_coef_max'    : 8.0,
                        'max_grad_norm_min'     : 1.0,
                        'max_grad_norm_max'     : 4.0,
                        'discount_min'          : 0.93,
                        'discount_max'          : 1.0,
                        'consistency_coef_min'  : 0.3,
                        'consistency_coef_max'  : 0.6
                      }

        # Set hyperparams
        self.learning_rate = self.params['learning_rate_max']*self.get_betavariate_value(1, 7) + self.params['learning_rate_min']
        self.hidden_nodes = int(self.get_uniform_value(self.params['h_nodes_layer_min'], self.params['h_nodes_layer_max']))
        if self.hidden_nodes % 2 == 0:
            self.hidden_nodes += 1
        self.base_xplr_rate = self.get_uniform_value(self.params['base_explr_min'], self.params['base_explr_max'])
        self.alpha = 1.25
        self.beta = 5.625
        self.accuracy_coef = self.get_uniform_value(self.params['acc_coef_min'], self.params['acc_coef_max'])
        self.advantage_coef = self.get_uniform_value(self.params['advantage_coef_min'], self.params['advantage_coef_max'])
        self.max_grad_norm = self.get_uniform_value(self.params['max_grad_norm_min'], self.params['max_grad_norm_max'])
        self.discount = self.get_uniform_value(self.params['discount_min'], self.params['discount_max'])
        self.consist_coef = self.get_uniform_value(self.params['consistency_coef_min'], self.params['consistency_coef_max'])

        # Append this to workerData folder so we know which seed obtained which results
        self.seed = seed

    # Available random functions
    def get_uniform_value(self, a, b):
        """Takes a & b and returns a value using uniform distribution between the two"""
        return random.uniform(a, b)

    def get_gammavariate_value(self, alpha, beta):
        """Takes alpha & beta and returns a value using gammavariate distribution between the two"""
        return random.gammavariate(alpha, beta)

    def get_betavariate_value(self, alpha, beta):
        """Takes alpha & beta and returns a value using betavariate distribution between the 0 and 1"""
        return random.betavariate(alpha, beta)

    def get_weibullvariate_value(self, alpha, beta):
        """Takes alpha & beta and returns a value using weibullvariate distribution where alpha is the scale and beta is the shape"""
        return random.betavariate(alpha, beta)


def main(start=None):
    if start == None:    
        for it in range(0,2):
            dump_params(it)
    else:
        dump_params(start)


def dump_params(seed):
    hp = HyperParams(seed)
    print("\nTesting Hyperparam Randomization:")
    print("Learning Rate:", hp.learning_rate)
    print("Hidden Nodes:", hp.hidden_nodes)
    print("Explore Rate Base:",hp.base_xplr_rate)
    print("Accuracy Coeff:", hp.accuracy_coef)
    print("Advantage Coeff:", hp.advantage_coef)
    print("Max Grad Norm:", hp.max_grad_norm)
    print("Discount:", hp.discount)
    print("Consistency Coeff:", hp.consist_coef)

if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except IndexError:
        main()