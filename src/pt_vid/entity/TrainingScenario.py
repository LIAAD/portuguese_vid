from pydantic import BaseModel, Field

class TrainingScenario(BaseModel):
    name: str