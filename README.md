# 🏡 California Housing Price Prediction API  

A production-ready Machine Learning REST API built with **FastAPI**, **Scikit-Learn**, and **Pandas** that serves a trained Random Forest Regression model to predict California median home values.

The API supports single-record JSON inference with strict schema validation via Pydantic, as well as high-throughput batch CSV processing with in-memory streaming responses.

---

## 🚀 Features

- **Single Prediction (`POST /predict`):** Validates geospatial boundaries (California latitude/longitude) and socioeconomic factors via Pydantic before scoring.
- **Batch CSV Processing (`POST /predict-file`):** Ingests raw tabular data, verifies required columns, appends predictions, and streams back the populated CSV without intermediate disk writes.
- **Service Monitoring (`GET /health`):** Exposes service health, model architecture, expected feature schema, and error benchmarks ($39,000 MAE).
- **Interactive Documentation:** Automatic Swagger UI (`/docs`) and ReDoc (`/redoc`) support out of the box.

---

## 📁 Project Structure

```text
california-house-prediction-api/
├── main.py                 # FastAPI application routes & prediction logic
├── train.py                # Model training and serialization script
├── housing_project.py      # Exploratory data analysis (EDA) script
├── house_model.joblib      # Serialized trained RandomForestRegressor
├── house_features.joblib   # Serialized feature column names
├── sample_test.csv         # Sample dataset to test batch predictions
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **Machine Learning:** Scikit-Learn (Random Forest Regressor)
- **Data Manipulation:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Data Validation:** Pydantic
- **ASGI Server:** Uvicorn

---

## 🏁 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/prekshajain-oss/California_Housing_Price_Prediction_API.git
cd California_Housing_Price_Prediction_API
```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API Server

```bash
uvicorn main:app --reload
```

The API will be live at `http://127.0.0.1:8000`.

- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 API Endpoints

### 1. Root & Health Check

- **`GET /`**: Welcome message and index of available endpoints.
- **`GET /health`**: Returns model status, architecture, and expected features.

### 2. Single House Prediction

- **Method:** `POST`
- **Path:** `/predict`
- **Content-Type:** `application/json`

**Sample Request Body:**
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0,
  "AveOccup": 2.5555,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

**Sample Response:**
```json
{
  "predicted_price": "$452,600",
  "raw_prediction": 4.526,
  "confidence_range": "$413,600 - $491,600"
}
```

### 3. Batch Prediction via CSV

- **Method:** `POST`
- **Path:** `/predict-file`
- **Content-Type:** `multipart/form-data`

Upload a `.csv` file containing the 8 required features. A file with appended prediction columns will be returned as an attachment.

**cURL Example:**
```bash
curl -X POST "http://127.0.0.1:8000/predict-file" \
  -H "accept: text/csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_test.csv" \
  --output predictions.csv
```

---

## 📊 Dataset & Feature Schema

The model is trained on the standard **California Housing Dataset** from `sklearn.datasets`:

| Feature | Description | Range / Validation |
| :--- | :--- | :--- |
| `MedInc` | Median income in block group | > 0 ($10,000s) |
| `HouseAge` | Median house age in block group | > 0 years |
| `AveRooms` | Average number of rooms per household | > 0 |
| `AveBedrms`| Average number of bedrooms per household | > 0 |
| `Population` | Block group population | > 0 |
| `AveOccup` | Average number of household members | > 0 |
| `Latitude` | Latitude coordinate | 32.0 to 42.0 |
| `Longitude`| Longitude coordinate | -125.0 to -114.0 |

---

## 🧠 Model Training

To retrain the model locally:

```bash
python train.py
```
This script will retrain the `RandomForestRegressor`, evaluate test metrics, and update `house_model.joblib`.
