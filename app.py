
import sys
import os

import certifi
import pymongo
import pandas as pd

from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.templating import Jinja2Templates

from uvicorn import run as app_run

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME
)


# --------------------------------------------------
# 1. Environment and MongoDB Configuration
# --------------------------------------------------

load_dotenv()

mongo_db_url = os.getenv("MONGODB_URL_KEY")

if not mongo_db_url:
    raise RuntimeError(
        "MONGODB_URL_KEY is missing. Add it to your .env file."
    )

ca = certifi.where()

client = pymongo.MongoClient(
    mongo_db_url,
    tlsCAFile=ca
)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]


# --------------------------------------------------
# 2. FastAPI Application Configuration
# --------------------------------------------------

app = FastAPI(
    title="Network Security - Phishing Detection",
    version="1.0.0",
    description="Machine Learning Based Phishing Website Detection"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")


# --------------------------------------------------
# 3. Homepage Route
# --------------------------------------------------

@app.get("/", tags=["authentication"])
async def index(request: Request):
    try:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={}
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# --------------------------------------------------
# 4. Model Training Route
# --------------------------------------------------

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()

        return Response(
            content="Training is successful",
            media_type="text/plain"
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# --------------------------------------------------
# 5. Prediction Route
# --------------------------------------------------

@app.post("/predict")
async def predict_route(
    request: Request,
    file: UploadFile = File(...)
):
    try:

        # Read uploaded CSV file
        df = pd.read_csv(file.file)

        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        # Load trained preprocessing object and model
        preprocessor = load_object(
            "final_model/preprocessor.pkl"
        )

        final_model = load_object(
            "final_model/model.pkl"
        )

        # Initialize prediction pipeline
        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=final_model
        )

        # Generate predictions
        y_pred = network_model.predict(df)

        # Convert predictions to integer labels
        prediction_series = pd.Series(
            y_pred,
            index=df.index
        ).astype(int)

        # Model target mapping:
        # Original -1 (Phishing) becomes 0
        # Original  1 (Legitimate) remains 1

        label_mapping = {
            0: "Phishing",
            1: "Legitimate"
        }

        # Check that all predictions have known labels
        unknown_labels = (
            set(prediction_series.unique())
            - set(label_mapping.keys())
        )

        if unknown_labels:
            raise ValueError(
                f"Unexpected prediction labels: {unknown_labels}"
            )

        # Add prediction columns
        df["predicted_column"] = prediction_series

        df["prediction_label"] = (
            prediction_series.map(label_mapping)
        )

        # Create prediction output folder
        output_folder = "prediction_output"

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        # Save predictions to CSV
        output_path = os.path.join(
            output_folder,
            "output.csv"
        )

        df.to_csv(
            output_path,
            index=False
        )

        # Calculate prediction summary
        total_records = len(df)

        phishing_count = int(
            (df["prediction_label"] == "Phishing").sum()
        )

        legitimate_count = int(
            (df["prediction_label"] == "Legitimate").sum()
        )

        # Convert DataFrame into HTML table
        table_html = df.to_html(
            index=False,
            classes="results-table",
            escape=True
        )

        logging.info(
            f"Prediction completed successfully. "
            f"Total: {total_records}, "
            f"Phishing: {phishing_count}, "
            f"Legitimate: {legitimate_count}"
        )

        # Display prediction results page
        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={
                "table": table_html,
                "total": total_records,
                "phishing_count": phishing_count,
                "legitimate_count": legitimate_count
            }
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

    finally:
        file.file.close()


# --------------------------------------------------
# 6. Download Prediction Results Route
# --------------------------------------------------

@app.get("/download")
async def download_predictions():
    try:
        output_path = os.path.abspath(
            os.path.join(
                "prediction_output",
                "output.csv"
            )
        )

        if not os.path.exists(output_path):
            return Response(
                content="No prediction results available. "
                        "Please upload a CSV file first.",
                status_code=404,
                media_type="text/plain"
            )

        return FileResponse(
            path=output_path,
            media_type="text/csv",
            filename="network_security_predictions.csv"
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# --------------------------------------------------
# 7. Application Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    app_run(
        app,
        host="0.0.0.0",
        port=8000
    )
