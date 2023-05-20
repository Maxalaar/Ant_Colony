from ray.rllib.models import ModelCatalog

from models.minimal_model import MinimalModel

ModelCatalog.register_custom_model('minimal_model', MinimalModel)
