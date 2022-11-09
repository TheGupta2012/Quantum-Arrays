class KDistinctness:
    def __init__(self, K) -> None:
        self.K = K

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, new_val):
        if not isinstance(new_val, int) or new_val <= 0:
            raise ValueError("K value must be an integer > 0")


class GeneralKDistinctness:
    def __init__(self, K, delta) -> None:
        self.K = K
        self.delta = delta

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, new_val):
        if not isinstance(new_val, int) or new_val <= 0:
            raise ValueError("K value must be an integer > 0")
