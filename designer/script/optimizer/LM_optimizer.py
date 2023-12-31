import sys
sys.path.append('./designer/script/')


from tmm.get_jacobi_n_adjoint import get_jacobi_free_form
from tmm.get_jacobi import get_jacobi_simple
from tmm.get_spectrum import get_spectrum_free, get_spectrum_simple

from optimizer.grad_helper import stack_f, stack_J, stack_init_params
from utils.loss import calculate_RMS_f_spec, rms
from spectrum import BaseSpectrum
from film import FreeFormFilm, TwoMaterialFilm
import numpy as np
from typing import Sequence
import copy
from optimizer.optimizer import GradientOptimizer
from abc import abstractmethod

"""adam.py - Adam optimizer for thin film properties.

This module contains the AdamOptimizer class, which implements the Adam optimization
algorithm for optimizing thin film properties. This class inherits from the Optimizer
class defined in optimizer.py.

(generated by chatGPT)
"""


class LMOptimizer(GradientOptimizer):
    """

    """

    def __init__(
        self,
        film,
        target_spec_ls: Sequence[BaseSpectrum],
        max_steps,
        **kwargs
    ):
        super().__init__(film, target_spec_ls, max_steps, **kwargs)

        # adam hyperparameters
        self.h_tol = 1e-5 if 'h_tol' not in kwargs else kwargs['h_tol']

        # initialize optimizer
        self.max_steps = max_steps
        self.max_patience = self.max_steps
        self.current_patience = self.max_patience
        self.best_loss = 0.
        self.nu = 2
        self.mu = 1  
        self.n_arrs_ls = stack_init_params(self.film, self.target_spec_ls)

        self._get_param()  # init variable x

        # allocate space for f and J
        self.J = np.empty((self.total_wl_num, self.x.shape[0]))
        self.f = np.empty(self.total_wl_num)

    def optimize(self):
        # in case not do_record, return [initial film], [initial loss]
        self._record()

        for self.i in range(self.max_steps):
            self._optimize_step()
            self._set_param()
            if self.is_recorded:
                self._record()
            if self.is_shown:
                self._show()
            if not self._update_best_and_patience():
                break
            if self._break_because_small_step():
                break
        self.x = self.best_x
        self._set_param()  # restore to best x
        return self._rearrange_record()

    def _validate_loss(self):
        # return rms(self.f) THIS IS WRONG! should calculate on val set
        return calculate_RMS_f_spec(self.film, self.target_spec_ls)

    def _optimize_step(self):
        self._mini_batching()  # make sgd params
        stack_f(
            self.f,
            self.n_arrs_ls,
            self.film.get_d(),
            self.target_spec_ls,
            get_f=self.get_f
        )
        stack_J(
            self.J,
            self.n_arrs_ls,
            self.film.get_d(),
            self.target_spec_ls,
            MAX_LAYER_NUMBER=250,  # TODO: refactor. This is not used
            get_J=self.get_J
        )

        # LM descent step. Including update x. 
        raise NotImplementedError

