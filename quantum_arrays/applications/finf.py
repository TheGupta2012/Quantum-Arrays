from ..algorithms.pmax import PMax

# essentially the finf is the pmax algo only
class Finf(PMax):
    def __init__(self, accuracy, error, algo_type) -> None:
        super().__init__(accuracy, error, algo_type)
