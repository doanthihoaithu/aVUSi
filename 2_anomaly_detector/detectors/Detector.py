import os
from abc import ABC, abstractmethod

from utils import get_project_root


class Detector(ABC):
    model = None
    config = None
    def __init__(self, customParameters, config):
        self.customParameters = customParameters
        self.config = config
        os.makedirs(self.absolute_dataOutputDir, exist_ok=True)
        os.makedirs(self.absolute_modelInputDir, exist_ok=True)

    @property
    @abstractmethod
    def name(self) -> str:
        pass


    @property
    def absolute_dataOutputDir(self) -> str:
        project_root_dir = get_project_root()
        return os.path.join(project_root_dir, self.config.dataOutput)



    @property
    def absolute_dataInputDir(self) -> str:
        project_root_dir = get_project_root()
        return os.path.join(project_root_dir, self.config.dataInput)


    @property
    def absolute_modelInputDir(self) -> str:
        project_root_dir = get_project_root()
        return os.path.join(project_root_dir, self.config.modelInputDir)

    @property
    def absolute_modelOutputDir(self) -> str:
        project_root_dir = get_project_root()
        return os.path.join(project_root_dir, self.config.modelOutputDir)

    @property
    def absolute_modelInputFile(self) -> str:
        project_root_dir = get_project_root()
        return os.path.join(project_root_dir, self.config.modelInputFile)

    @property
    def absolute_modelOutputFile(self) -> str:
        project_root_dir = get_project_root()
        return os.path.join(project_root_dir, self.config.modelOutputFile)
    @abstractmethod
    def _init_model(self):
        pass
    @abstractmethod
    def train(self, data, labels=None)-> float:
        pass

    @abstractmethod
    def predict(self, test_dict) -> dict:
        pass