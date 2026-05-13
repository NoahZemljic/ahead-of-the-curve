import logging

from data_loader import TrainingDataLoader
from preprocessing import PreProcessor
from train_models import Trainer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline: load, preprocess, train, evaluate."""

    def __init__(self):
        """Initialize the training pipeline components."""
        self.data_loader = TrainingDataLoader()
        self.preprocessor = PreProcessor()
        self.trainer = Trainer()

    def run(self):
        """Load labelled data, preprocess it, train models, and run evaluation/promotion."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        logger.info("Loading labelled data from Hopsworks")
        mature_models = self.data_loader.load()
        if mature_models.empty:
            logger.warning("No labelled data available, exiting")
            return

        logger.info(f"Loaded {len(mature_models)} rows")
        data = self.preprocessor.process(mature_models)

        self.trainer.train(data)
        logger.info("Training pipeline complete")


if __name__ == "__main__":
    TrainingPipeline().run()
