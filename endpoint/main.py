import torch
from ftlangdetect import detect
from transformers import pipeline
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException, status
from pt_vid.data.Delexicalizer import Delexicalizer

print(f"Run on {torch.cuda.get_device_name()}")

app = FastAPI()

@app.on_event("startup")
def load_pipeline():
    torch.cuda.empty_cache()    
    app.delexicalized_pipeline = pipeline("text-classification", model="liaad/PtVId")
    app.delexicalizer = Delexicalizer(prob_pos_tag=0.6, prob_ner_tag=0.0)

@app.on_event("shutdown")
def delete_pipeline():
    del app.delexicalized_pipeline
    del app.delexicalizer

    torch.cuda.empty_cache()

### Requests ###

class SimilarityRequest(BaseModel):
    raw_text: str
    scenario: Literal["bert-delexicalized", "bert-raw", "n-grams-delexicalized", "n-grams-raw"]  

### Response ###
class FastTextResponse(BaseModel):
    lang: str
    confidence: float
class SimilarityResponse(BaseModel):
    raw_text: str
    delexicalized_text: str
    european_portuguese: float
    brazilian_portuguese: float
    other_languages: Optional[List[FastTextResponse]] = Field(
        default=None,
        title="List of other languages",
        description="If fasttext reports other languages, they will be listed here. This field is only present if there is error in the prediction."
    )

    

### Code ###
@app.post("/detect", response_model=SimilarityResponse, responses={
    "400": {"description": "Text is not in Portuguese"},
    "501": {"description": "This scenario is not implemented yet"}
})
async def root(request: SimilarityRequest)-> SimilarityResponse:
    if request.scenario != 'bert-delexicalized':
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="This scenario is not implemented yet")
    else:
        pipeline = app.delexicalized_pipeline

    result = detect(request.raw_text)

    if not result or result.get('lang', None) != "pt":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text is not in Portuguese")
    
    response = pipeline(request.raw_text)
    score = response[0]['score']
    is_european = response[0]['label'] == "PT-PT"

    return SimilarityResponse(
        raw_text=request.raw_text,
        delexicalized_text=app.delexicalizer.delexicalize(request.raw_text),
        european_portuguese=score if is_european else 1 - score,
        brazilian_portuguese=1 - score if is_european else score
    )
