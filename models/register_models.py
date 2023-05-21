from ray.rllib.models import ModelCatalog

from models.minimal_model import MinimalModel
from models.minimal_lstm_model import MinimalLSTMModel

ModelCatalog.register_custom_model('minimal_model', MinimalModel)
ModelCatalog.register_custom_model('minimal_lstm_model', MinimalLSTMModel)