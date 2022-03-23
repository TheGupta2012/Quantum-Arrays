import math
from .high_dist import HighDist


class PMax:
    def __init__(self, accuracy, error, algo_type) -> None:
        """ """
        self.accuracy = accuracy
        self.error = error
        self.algo_type = algo_type

        precision = self._get_bit_precision(
            self.accuracy
        )  # find the precision through the accuracy
        self._high_dist = HighDist(0.5, self.error, precision)

    @property
    def accuracy(self):
        return self._accuracy

    @accuracy.setter
    def accuracy(self, new_val):
        self._check_param("Accuracy", new_val)
        self._accuracy = new_val

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, new_val):
        self._check_param("Error", new_val)
        self._error = new_val

    def _check_param(self, param, val):
        if not isinstance(val, float) or val <= 0 or val >= 1:
            raise TypeError(f"{param} must be a float value in range (0,1)")

    @property
    def algo_type(self):
        return self._algo_type

    @algo_type.setter
    def algo_type(self, new_val):

        if new_val not in set("additive", "relative"):
            raise ValueError(
                'The type of error can only be one of ["additive", "relative"]'
            )

        self._algo_type = new_val

    # initializers
    def _init_params(self, N):
        # boundaries
        self._lower = 1 / N
        self._upper = 1

        self._set_K()

        if self._algo_type == "additive":
            threshold = 0.5
            updater = self.accuracy / 4
        else:
            threshold = (1 - self._updater) ** (2 ** (self._K - 1))
            updater = 1 - math.sqrt(1 - self.accuracy)

        self._threshold = threshold
        self._updater = updater

    def _set_K(self, N=None):

        k = 1
        if self.algo_type == "additive":
            target = self.accuracy / 2
            while True:
                if 2 ** (-k) > target:
                    k += 1
                else:
                    break

        else:
            target = 1 / N
            # try to increase for right
            while True:
                right_val = (1 - self.accuracy) ** (2 ** (k - 2))
                if right_val > target:
                    k += 1
                else:
                    k -= 1
                    break
        self._K = k

    def _update_threshold(self, i_val, increase):
        if self.algo_type == "additive":
            update = 2 ** (-(i_val + 1))
            if increase:
                self._threshold += update
            else:
                self._threshold -= update
        else:
            exponent = 2 ** (self._K - (i_val + 1))
            update = (1 - self._updater) ** exponent
            if increase:
                self._threshold /= update
            else:
                self._threshold *= update

    def _get_bit_precision(self, val):
        # return how many bits would
        # allows us to encode this level
        # of accuracy
        pass

    # main methods
    def encode(self, array, M, N=None):
        """_summary_

        Args:
            array (_type_): _description_
            M (_type_): _description_
            N (_type_, optional): _description_. Defaults to None.
        """
        # will need HDist object for the encoding
        # and the running of the algorithm
        self._high_dist.encode(array, M, N)

    def _init_check(self):
        if self.algo_type == "additive":
            return self._threshold <= self.accuracy
        else:
            return self._threshold <= (1 - self._updater)

    def run(self, backend, shots, optimization_level):
        """_summary_"""
        # this again uses the HDist object

        for i in range(1, self._K + 1):
            if self._init_check():
                if self.algo_type == "additive":
                    self._upper = self.accuracy
                else:
                    self._upper = 1 - self._updater
                break

            else:
                self._high_dist.run(
                    backend=backend,
                    shots=shots,
                    optimization_level=optimization_level,
                    p_max_type=self.algo_type,
                    p_max_init=(i == 1),
                )
                result = self._high_dist.get_result()
                increase = result["algorithm_result"]

                self._update_threshold(i, increase)

                new_precision = self._get_bit_precision(self._threshold * self._updater)
                self._high_dist._update_params(self._threshold, new_precision)

        if self.algo_type == "additive":
            result = [self._lower, self._upper]
        else:
            result = [self._upper]
