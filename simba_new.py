import numpy as np
from scipy.optimize import OptimizeResult
from EAreal import ea_real
from scipy.optimize.optimize import _status_message


__all__ = ['simba']

_MACHEPS = np.finfo(np.float64).eps


def simba_single(net, limits, image, label, maxfev=30000):

    solver = SimBA(net=net, limits=limits, maxfev=maxfev)
    return solver.simba_single(image, label)
    # Include the net for calculating probabilities

class SimBA(object):
    def __init__(self, net, limits, maxfev=30000):
        self.maxfev = maxfev
        self.pixels = 1
        # Parameters of SimBA, if the input is a batch
        self.batch = False 
        self.net = net
        self.epsilon = 0.2
        self.targeted = False
        self.limits = limits
        self.parameter_count = np.size(self.limits, 1)

            
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
            
            
        def simba_single(self, image, label):
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
                    
