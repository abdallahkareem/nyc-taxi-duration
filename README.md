# NYC Taxi Trip Duration Prediction

An end-to-end machine learning project for predicting NYC taxi trip duration, with a focus on building a reproducible and maintainable ML pipeline following MLOps practices.

## Pipeline

```
Data → Preprocessing → Feature Engineering → Training
                                      ↓
                              Experiment Tracking
                                      ↓
                              Model Evaluation
                                      ↓
                         Docker → CI → Deployment
```

## MLOps

| Practice | Implementation |
|---|---|
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Reproducibility | DVC + locked dependencies |
| Testing | Pytest |
| Code Quality | Pre-commit |
| Containerization | Docker |
| CI | GitHub Actions |

## Data

The dataset contains NYC taxi trips with information such as pickup/dropoff locations, passenger count, and timestamps.

A chronological 80/20 split is used to evaluate the model on later trips and reduce the risk of temporal data leakage.

## Model Evaluation

| Metric | Result |
|---|---|
| MAE | — |
| RMSE | — |
| R² | — |

## Running the Project

```bash
git clone https://github.com/abdallahkareem/nyc-taxi-duration.git
cd nyc-taxi-duration

uv sync
dvc repro
pytest
```

Run with Docker:

```bash
docker compose up --build
```

## Project Structure

```
src/          # ML pipeline
data/         # Dataset
models/       # Model artifacts
tests/        # Tests
config/       # Configuration
notebooks/    # Experiments
```
