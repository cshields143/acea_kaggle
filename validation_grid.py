from mods.models import AlwaysZeroPredictor, SimpleLinearPredictor
from mods.validate import validate_model
import os
import warnings
warnings.filterwarnings('ignore')

def initialize_folders(base):
    os.mkdir(base)
    os.mkdir(f"{base}aquifer/")
    os.mkdir(f"{base}waterspring/")
    os.mkdir(f"{base}river/")
    os.mkdir(f"{base}lake/")

for chunk in [5,10,15]:
    for model in [AlwaysZeroPredictor,SimpleLinearPredictor]:
        initialize_folders(f"data/models/{model.name}_{chunk}/")
        validate_model('data/models/', chunk, model, 'data/raw/')
