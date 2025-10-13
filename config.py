import yaml
from pydantic import BaseModel

with open('setting.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)


class Settings(BaseModel):
    model_path: str = config_dict['model_path']
