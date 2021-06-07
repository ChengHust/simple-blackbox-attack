import numpy as np
from scipy.optimize import OptimizeResult
from EAreal import ea_real
from scipy.optimize.optimize import _status_message


__all__ = ['slsmof']

_MACHEPS = np.finfo(np.float64).eps


def simba_new(func, net, limits, popsize=100, sub_popsize=10, topk=10, maxfev=30000,
           callback=None, polish=False, init='latinhypercube'):

    solver = SimBA(func, net, limits, popsize=popsize, sub_popsize=sub_popsize, topk=topk,
                       maxfev=maxfev, callback=callback, polish=polish, init=init)

    # Include the net for calculating probabilities

class SimBA(object):
    def __init__(self, net, func, limits, popsize=100, sub_popsize=10, topk=10, maxfev=30000,
                 callback=None, polish=False, init='latinhypercube'):
        self.popsize = popsize
        self.maxfev = maxfev
        self.evaluated = 0
        self.best = None
        self.pixels = 1

        # Parameters of SimBA, if the input is a batch
        self.batch = False 
        self.net = net
        self.epsilon = 0.2
        self.targeted = False
        
        
        
        self.func = func
        self.callback = callback
        self.polish = polish

        self.limits = limits
        self.parameter_count = np.size(self.limits, 1)

        self.population_shape = (self.popsize,
                                 self.parameter_count)

        # initialization
        if init == 'latinhypercube':
            decs = self.lhsampling(self.popsize, self.parameter_count) * (
                self.limits[1] - self.limits[0]) + self.limits[0]

            self.population = (decs, self.func(decs))
        else:
            decs = np.random.random((self.popsize, self.parameter_count)) * (
                self.limits[1] - self.limits[0]) + self.limits[0]
            self.population = (decs, self.func(decs))
            
        def _perturb(self, image, delta):
            delta = np.array(delta)
            if len(delta.shape) < 2:
                delta = np.array([delta])
            num_delta = len(delta)
            adv_image = image.clone().detach().to(self.net.device)
            adv_images = torch.cat([adv_image]*num_delta, dim=0)
            for idx in range(num_delta):
                pixel_info = delta[idx].reshape(self.pixels, -1)
                for pixel in pixel_info:
                    pos_x, pos_y = pixel[:2]
                    channel_v = pixel[2:]
                    for channel, v in enumerate(channel_v):
                        # delta is the difference instead of the final result
                        adv_images[idx, channel, int(pos_x), int(pos_y)] -= v
            return adv_images
            
            
        def simba_single(self, image, label, targeted= False):
            d = self.parameter_count
            diff = np.zeros(d)
            # SimBA assues that d >> self.maxfev
            perm = np.random.permutation(d)  
            if d <= self.maxfev:
                self.maxfev = d
            last_prob = self.net._get_prob(image)[:, label]
            self.evaluated += 1
            for i in range(self.maxfev):
                diff[perm[i]] = self.epsilon
                adv_image = self._perturb(image, diff)
                left_prob = self.net._get_prob(adv_image)[:, label]
                self.evaluated += 1
                if self.targeted != (left_prob < last_prob):
                    image = adv_image
                    last_prob = left_prob
                else:
                    adv_image = self._perturb(image, -diff)
                    right_prob = self.net._get_prob(adv_image)[:, label]
                    self.evaluated += 1
                    if self.targeted != (right_prob < last_prob):
                        image = adv_image
                        last_prob =  right_prob
            return image
                    
            
            
            
            
            
            
            
            
            

    def solve(self, net):
        population = self.population
        self.evaluated += self.popsize
        self.best = [self.evaluated, np.min(population[1])]

        gen = 1
        iteration = 0
        while self.evaluated <= self.maxfev:
            diff = np.zeros(d)
            diff[perm[i]] = self.epsilon
