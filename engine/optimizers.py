# ######################################################################
# Author: Cameron M. Gattis
# Created: 2026-06-06
# Updated: New.
# Purpose: Optimizer implementations for DiscoveryModel training.
#          Provides gradient-based weight update methods including
#          Stochastic Gradient Descent (SGD) and Adam optimization.
# ######################################################################

import numpy as np


class BaseOptimizer:
    # ######################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-06
    # Updated: New.
    # Purpose: Base optimizer interface.
    # ######################################################################

    def update(self, weights, gradients):
        '''
            Abstract optimizer update method.

            @params weights : numpy array
                    gradients : numpy array

            @return weights : numpy array

            @exception NotImplementedError : optimizer not implemented

            @notes Must be overridden by derived optimizer classes.
        '''

        raise NotImplementedError


class SGD(BaseOptimizer):
    # ######################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-06
    # Updated: New.
    # Purpose: Stochastic Gradient Descent optimizer.
    # ######################################################################

    def __init__(self, lr=0.001):
        '''
            Function initializer.

            @params lr : float

            @notes Learning rate controls update step size.
        '''

        self.lr = lr

    def update(self, weights, gradients):
        '''
            Performs SGD weight update.

            @params weights : numpy array
                    gradients : numpy array

            @return weights : numpy array

            @notes Applies:
                   weights = weights - lr * gradients
        '''

        return (weights - self.lr * gradients)


class Adam(BaseOptimizer):
    # ######################################################################
    # Author: Cameron M. Gattis
    # Created: 2026-06-06
    # Updated: New.
    # Purpose: Adaptive Moment Estimation optimizer.
    # ######################################################################

    def __init__(
            self,
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8
    ):
        '''
            Function initializer.

            @params lr : float
                    beta1 : float
                    beta2 : float
                    eps : float

            @notes Adam uses first and second moment estimates
                   to adapt learning rates during optimization.
        '''

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, gradients):
        '''
            Performs Adam weight update.

            @params weights : numpy array
                    gradients : numpy array

            @return weights : numpy array

            @notes Uses bias-corrected first and second moment
                   estimates for parameter updates.
        '''

        if self.m is None:
            self.m = np.zeros_like(gradients)
            self.v = np.zeros_like(gradients)

        self.t += 1
        self.m = (self.beta1 * self.m + (1 - self.beta1) * gradients)
        self.v = (self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2))
        m_hat = ( self.m / (1 - self.beta1 ** self.t))
        v_hat = (self.v / (1 - self.beta2 ** self.t))

        return (weights - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))