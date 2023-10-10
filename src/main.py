from fastapi import FastAPI, UploadFile
import io
import pandas as pd
from src.utils import utils

app = FastAPI()


@app.get("/home")
def read_root():
    return {"message": "This is home"}



@app.post("/predict/")
async def predict(beneficiary: UploadFile, inpatient: UploadFile, outpatient: UploadFile):
    if (
            beneficiary.filename.endswith(".csv")
            and inpatient.filename.endswith(".csv")
            and outpatient.filename.endswith(".csv")
    ):
        beneficiary_contents = await beneficiary.read()
        inpatient_contents = await inpatient.read()
        outpatient_contents = await outpatient.read()

        beneficiary_df = pd.read_csv(io.BytesIO(beneficiary_contents))
        inpatient_df = pd.read_csv(io.BytesIO(inpatient_contents))
        outpatient_df = pd.read_csv(io.BytesIO(outpatient_contents))

        df, provider_id = utils.preporcessing(beneficiary_df, inpatient_df, outpatient_df)
        result_df = pd.concat([df, provider_id], axis=1)

        df['Prediction'] = df['Prediction'].map({0: 'No', 1: 'Yes'})

        return result_df[['Prediction', 'Provider']].to_dict(orient='records')

