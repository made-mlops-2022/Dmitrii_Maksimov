from typing import Literal
from pydantic import BaseModel, validator
from fastapi.exceptions import HTTPException


class HeartDiseaseResponse(BaseModel):
    condition: Literal[0, 1]


class HeartDiseaseModel(BaseModel):
    age: float
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: float
    chol: float
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]

    @validator("age")
    def validation_age(cls, val):
        if val < 0 or val > 150:
            raise HTTPException(
                detail=[{"msg": f"ValueError: age value {val}"}], status_code=400
            )
        return val

    @validator("trestbps")
    def validation_trestbps(cls, val):
        if val < 0 or val > 500:
            raise HTTPException(
                detail=[{"msg": f"ValueError: trestbps value {val}"}], status_code=400
            )
        return val

    @validator("chol")
    def validation_chol(cls, val):
        if val < 0 or val > 500:
            raise HTTPException(
                detail=[{"msg": f"ValueError: chol value {val}"}], status_code=400
            )
        return val

    @validator("thalach")
    def validation_thalach(cls, val):
        if val < 0 or val > 300:
            raise HTTPException(
                detail=[{"msg": f"ValueError: thalach value {val}"}], status_code=400
            )
        return val

    @validator("oldpeak")
    def validation_oldpeak(cls, val):
        if val < 0 or val > 10:
            raise HTTPException(
                detail=[{"msg": f"ValueError: oldpeak value {val}"}], status_code=400
            )
        return val
