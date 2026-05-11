from data_loader import TrainingDataLoader

from preprocessing import PreProcessor

class TrainingPipeline:
    def __init__(self):
        self._data_loader = TrainingDataLoader()
        self._preprocessor = PreProcessor()

    def run(self):
        mature_models = self._data_loader.load()

        splits = self._preprocessor.process(mature_models)
        splits["X_train"].to_csv('X_train.csv')



if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
