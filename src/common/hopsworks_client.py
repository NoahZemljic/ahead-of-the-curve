import logging
import os

import hopsworks
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class HopsworksFeatureStoreClient:
    """Shared Hopsworks feature store connection manager.

    Encapsulates the low-level connection logic so that both the feature
    pipeline and the training pipeline can reuse it.
    """

    FEATURE_GROUP_NAME = "frontier_models_features"
    FEATURE_GROUP_VERSION = 2

    def __init__(self):
        self._fs = None

    def get_feature_store(self):
        """Connect to Hopsworks and return the feature store handle (cached)."""
        if self._fs is None:
            api_key = os.getenv("HOPSWORKS_API_KEY")
            if not api_key:
                raise ValueError("HOPSWORKS_API_KEY environment variable not set")
            project = hopsworks.login(api_key_value=api_key)
            self._fs = project.get_feature_store()
            logger.info("Connected to Hopsworks feature store")
        return self._fs
