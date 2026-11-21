# Network Security – Phishing Website Detection

**Adaptation maintained by: Priti Kumari**

This repository is an adapted copy of an existing educational Network Security / phishing-detection project. The upstream implementation was originally attributed to Krish Naik. Priti Kumari is identified as the maintainer of this adaptation; this does not claim sole authorship of the original codebase. Check the upstream repository's licence and follow its attribution and redistribution conditions before publishing.

## Project contents

The original project layout is retained, including:

- `networksecurity/` – ingestion, validation, transformation, model training, pipeline, utilities and entities
- `Network_Data/` – source phishing dataset
- `data_schema/` – schema definition
- `Artifacts/` – existing timestamped pipeline outputs
- `final_model/` – saved model and preprocessing objects
- `logs/` – existing run logs
- `prediction_output/` – sample prediction output
- `templates/` – FastAPI HTML template
- `app.py`, `main.py`, `push_data.py`, `test_mongodb.py`
- `Dockerfile` and `.github/workflows/main.yml`

## Setup (Windows CMD / VS Code)

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure MongoDB Atlas

Copy the template and edit it locally:

```cmd
copy .env.example .env
notepad .env
```

Set `MONGODB_URL_KEY` to your MongoDB Atlas connection string. The configured database is `PRITI_NETWORK_SECURITY` and the collection is `NetworkData`. Add your current public IP to Atlas Network Access and make sure your database user has the necessary permissions.

**Never commit `.env` or paste database credentials into source files.** The `.gitignore` excludes `.env`.

## Test MongoDB

```cmd
python test_mongodb.py
```

## Upload the dataset

```cmd
python push_data.py
```

## Run the training pipeline

```cmd
python main.py
```

## Run the API

```cmd
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs`. The prediction endpoint expects a CSV containing the project's expected pre-extracted features.

## MLflow tracking

By default, the model trainer uses MLflow's local tracking location. To use your own remote MLflow service, set `MLFLOW_TRACKING_URI` and (if required) `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` in your local `.env`. No third-party account or access token is embedded in this copy.

## GitHub

Open this folder in VS Code, review the staged files with `git status`, and confirm `.env` is not staged before committing. Add your own GitHub repository as the remote rather than using another contributor's account.
