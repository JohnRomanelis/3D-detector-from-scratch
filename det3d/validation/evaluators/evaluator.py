from abc import ABCMeta, abstractmethod

class Evaluator(metaclass=ABCMeta):
    """
        Base class to evaluate a model on a dataset.

        TODO: Add support for:
            - Save detection results to a file
            - Evaluate from file
            - Evalute without saving a file 
    """

    @abstractmethod
    def save_to_file(self):
        pass

    @abstractmethod
    def read_detections_from_file(self, filename):
        pass

    @abstractmethod
    def evaluate(self, detections):
        pass

    @abstractmethod
    def read_ground_trurh(self, ground_truth_dir):
        pass

