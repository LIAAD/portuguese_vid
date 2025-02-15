from pt_vid.entity.Entity import Entity

class TestResult(Entity):
    model: object
    p_pos: float
    p_ner: float
    f1_score: float
    accuracy: float