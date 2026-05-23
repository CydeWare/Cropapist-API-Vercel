from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import requests
from datetime import date, timedelta
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
from typing import Optional
import re


stats_df = pd.read_csv("yield_stats_per_crop.csv", index_col="Item")

with open("yield_percentiles_per_crop.pkl", "rb") as f:
    percentile_data = pickle.load(f)

model = joblib.load("yield_model.pkl")
label_encoder = joblib.load("item_label_encoder.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    latitude: float
    longitude: float
    item: str


class PredictResponse(BaseModel):
    predicted_yield: float
    yield_category: str
    yield_percentile: Optional[float]
    average_rain_fall_mm_per_year: float
    avg_temp: float
    year: int
    warning: Optional[str]


def classify_yield_per_crop(crop: str, predicted_yield: float):
    if crop not in stats_df.index or crop not in percentile_data:
        return "unknown_crop", None

    row = stats_df.loc[crop]
    q1 = row["q1"]
    q3 = row["q3"]

    if predicted_yield < q1:
        category = "bad"
    elif predicted_yield > q3:
        category = "good"
    else:
        category = "medium"

    values = percentile_data[crop]["values"]
    n = percentile_data[crop]["n"]

    pos = np.searchsorted(values, predicted_yield, side="right")
    percentile = (pos / n) * 100.0

    return category, percentile


def fetch_seasonal_with_auto_limit(base_url, lat, lon, start_date, end_date):
    def build_url(e_date):
        return (
            base_url
            + f"?latitude={lat}&longitude={lon}"
            + f"&start_date={start_date}&end_date={e_date}"
            + "&daily=rain_sum,temperature_2m_mean&timezone=auto"
        )

    warning = None

    url = build_url(end_date)
    response = requests.get(url)

    if response.status_code == 200:
        return response.json(), end_date, None

    try:
        error_json = response.json()
        reason = error_json.get("reason", "")

        matches = re.findall(r"\d{4}-\d{2}-\d{2}", reason)

        if len(matches) >= 2:
            max_date = matches[-1]
            warning = None

            retry_url = build_url(max_date)
            retry_response = requests.get(retry_url)

            if retry_response.status_code == 200:
                return retry_response.json(), max_date, warning

    except Exception:
        pass

    return None, None, "Seasonal forecast unavailable for this location or date."


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/health")
def health_check():
    return {"message": "API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict_yield(payload: PredictRequest):
    today = date.today()
    year = today.year

    if payload.longitude < -180 or payload.longitude > 180 or payload.latitude < -90 or payload.latitude > 90:
        return PredictResponse(
            predicted_yield=0.0,
            yield_category="invalid_location",
            yield_percentile=None,
            average_rain_fall_mm_per_year=0.0,
            avg_temp=0.0,
            year=year,
            warning="Invalid coordinates."
        )

    first_day_of_year = date(year, 1, 1)
    today_minus_4 = today - timedelta(days=4)
    last_day_of_year = date(year, 12, 31)

    first_day_str = first_day_of_year.isoformat()
    today_minus_4_str = today_minus_4.isoformat()
    last_day_str = last_day_of_year.isoformat()

    lat = payload.latitude
    lon = payload.longitude

    archive_url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={first_day_str}&end_date={today_minus_4_str}"
        "&daily=rain_sum,temperature_2m_mean&timezone=auto"
    )

    archive_response = requests.get(archive_url)
    archive_response.raise_for_status()
    archive_json = archive_response.json()

    seasonal_base = "https://seasonal-api.open-meteo.com/v1/seasonal"

    seasonal_json, seasonal_used_end, seasonal_warning = fetch_seasonal_with_auto_limit(
        seasonal_base,
        lat,
        lon,
        today_minus_4_str,
        last_day_str
    )

    archive_rain = archive_json["daily"]["rain_sum"]
    archive_temp = archive_json["daily"]["temperature_2m_mean"]

    seasonal_rain = []
    seasonal_temp = []

    if seasonal_json:
        seasonal_rain = seasonal_json["daily"]["rain_sum"]
        seasonal_temp = seasonal_json["daily"]["temperature_2m_mean"]

    all_rain = [x for x in (archive_rain + seasonal_rain) if x is not None]
    all_temp = [x for x in (archive_temp + seasonal_temp) if x is not None]


    average_rain_fall_mm_per_year = float(sum(all_rain))
    avg_temp_year = float(sum(all_temp) / len(all_temp)) if all_temp else 0.0

    warning_msg = seasonal_warning

    if avg_temp_year < 0:
        warning_msg = "Average yearly temperature below 0°C. Prediction may be unreliable."
    elif average_rain_fall_mm_per_year < 50:
        warning_msg = "Yearly rainfall extremely low. Prediction may be unreliable."
    elif avg_temp_year > 40:
        warning_msg = "Yearly temperature extremely high. Prediction may be unreliable."

    try:
        item_encoded = label_encoder.transform([payload.item])[0]
    except ValueError:
        return PredictResponse(
            predicted_yield=0.0,
            yield_category="unknown_item",
            yield_percentile=None,
            average_rain_fall_mm_per_year=average_rain_fall_mm_per_year,
            avg_temp=avg_temp_year,
            year=year,
            warning=warning_msg
        )

    features = np.array([[
        average_rain_fall_mm_per_year,
        avg_temp_year,
        year,
        lat,
        lon,
        item_encoded
    ]])

    pred = model.predict(features)[0]
    category, percentile = classify_yield_per_crop(payload.item, pred)

    return PredictResponse(
        predicted_yield=float(pred),
        yield_category=category,
        yield_percentile=percentile,
        average_rain_fall_mm_per_year=average_rain_fall_mm_per_year,
        avg_temp=avg_temp_year,
        year=year,
        warning=warning_msg,
    )
