import logging

from data_loader import TrainingDataLoader
from deploy import GCPDeployer
from preprocessing import PreProcessor
from train_models import Trainer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline: load, preprocess, train, evaluate, deploy."""

    def __init__(self):
        """Initialize the training pipeline components."""
        self.data_loader = TrainingDataLoader()
        self.preprocessor = PreProcessor()
        self.trainer = Trainer()
        self.deployer = GCPDeployer()

    def run(self):
        """Load labelled data, preprocess it, train models, and run evaluation/promotion/deployment."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        logger.info("Loading labelled data from Hopsworks")
        mature_models = self.data_loader.load()
        if mature_models.empty:
            logger.warning("No labelled data available, exiting")
            return

        logger.info(f"Loaded {len(mature_models)} rows")
        data = self.preprocessor.process(mature_models)

        promoted_models = self.trainer.train(data)

        if promoted_models:
            for model_info in promoted_models:
                self.deployer.deploy(**model_info)
        else:
            logger.info("No models promoted, skipping deployment")

        logger.info("Training pipeline complete")


if __name__ == "__main__":
    TrainingPipeline().run()
