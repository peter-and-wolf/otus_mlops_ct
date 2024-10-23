import logging
# Switch off the annoying mlflow logs
logger = logging.getLogger("mlflow")
logger.setLevel(logging.CRITICAL)


ARTIFACT_PATH = 'sk_models'
MODEL_NAME = 'otus'
PROD_ALIAS = 'Prod'
CHAMP_ALIAS='Champion'
TRAINING_EXPERIMENT_NAME = 'training'
EVALUATION_EXPERIMENT_NAME = 'evaluation'
