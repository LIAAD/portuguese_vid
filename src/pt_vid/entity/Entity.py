from abc import ABC
from pydantic import BaseModel

class Entity(BaseModel, ABC):
    pass