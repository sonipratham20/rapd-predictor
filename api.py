"""
FastAPI service for the US-Accidents severity model.

Setup:
    pip install fastapi uvicorn joblib pandas numpy scikit-learn xgboost lightgbm

Put these in the same folder:
    api.py
    LightGBM.pkl                 (or XGBoost.pkl / Random_Forest.pkl)

Run:
    uvicorn api:app --reload
Docs:
    http://127.0.0.1:8000/docs
"""

import os
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAL = ["Sunrise_Sunset", "Weather_Condition", "State"]


class BasicPreprocessor:
    """Same class used in training, so joblib can rebuild final_processor."""

    def __init__(self, features):
        self.features = features
        self.cat_cols = [c for c in features if c in CATEGORICAL]
        self.num_cols = [c for c in features if c not in CATEGORICAL]
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
        )

    def fit(self, X):
        if self.num_cols:
            self.imputer.fit(X[self.num_cols])
        if self.cat_cols:
            self.encoder.fit(X[self.cat_cols].fillna("Unknown"))
        return self

    def transform(self, X):
        parts = []
        if self.num_cols:
            num = self.imputer.transform(X[self.num_cols])
            parts.append(np.asarray(num, dtype=np.float32))
        if self.cat_cols:
            cat = self.encoder.transform(X[self.cat_cols].fillna("Unknown"))
            parts.append(np.asarray(cat, dtype=np.float32))
        return np.ascontiguousarray(np.hstack(parts), dtype=np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# The pkl may have been saved from a notebook (class lives in __main__) or from
# train_accident_severity.py. Register the class under both names.
import sys
import types
import __main__

__main__.BasicPreprocessor = BasicPreprocessor
_shim = types.ModuleType("train_accident_severity")
_shim.BasicPreprocessor = BasicPreprocessor
sys.modules.setdefault("train_accident_severity", _shim)

MODEL_PATH = os.getenv("MODEL_PATH", "LightGBM.pkl")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
processor = artifact["final_processor"]
features = artifact["features"]

# Compatibility patch: the pkl was saved with an older scikit-learn whose
# SimpleImputer lacks _fill_dtype, which newer versions require.
_imp = getattr(processor, "imputer", None)
if _imp is not None and not hasattr(_imp, "_fill_dtype"):
    _imp._fill_dtype = getattr(_imp, "_fit_dtype", np.dtype("float32"))

SEVERITY_LABELS = [1, 2, 3, 4]

app = FastAPI(title="US Accidents Severity API", version="1.0")


class AccidentInput(BaseModel):
    # Any field left out is imputed (median / "Unknown"), same as in training.
    start_time: Optional[datetime] = Field(None, description="ISO format, e.g. 2023-01-15T08:30:00")
    end_time: Optional[datetime] = Field(None, description="Only needed if the model uses Duration_Min")

    start_lat: Optional[float] = None
    start_lng: Optional[float] = None
    end_lat: Optional[float] = None
    end_lng: Optional[float] = None
    distance_mi: Optional[float] = None

    temperature_f: Optional[float] = None
    wind_chill_f: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_in: Optional[float] = None
    visibility_mi: Optional[float] = None
    wind_speed_mph: Optional[float] = None
    precipitation_in: Optional[float] = None

    sunrise_sunset: Optional[str] = Field(None, description="Day or Night")
    weather_condition: Optional[str] = None
    state: Optional[str] = Field(None, description="Two-letter code, e.g. CA")

    amenity: Optional[bool] = None
    bump: Optional[bool] = None
    crossing: Optional[bool] = None
    give_way: Optional[bool] = None
    junction: Optional[bool] = None
    no_exit: Optional[bool] = None
    railway: Optional[bool] = None
    roundabout: Optional[bool] = None
    station: Optional[bool] = None
    stop: Optional[bool] = None
    traffic_calming: Optional[bool] = None
    traffic_signal: Optional[bool] = None
    turning_loop: Optional[bool] = None


# API field name -> dataset column name used in training
COLUMN_MAP = {
    "start_lat": "Start_Lat",
    "start_lng": "Start_Lng",
    "end_lat": "End_Lat",
    "end_lng": "End_Lng",
    "distance_mi": "Distance(mi)",
    "temperature_f": "Temperature(F)",
    "wind_chill_f": "Wind_Chill(F)",
    "humidity_pct": "Humidity(%)",
    "pressure_in": "Pressure(in)",
    "visibility_mi": "Visibility(mi)",
    "wind_speed_mph": "Wind_Speed(mph)",
    "precipitation_in": "Precipitation(in)",
    "sunrise_sunset": "Sunrise_Sunset",
    "weather_condition": "Weather_Condition",
    "state": "State",
    "amenity": "Amenity",
    "bump": "Bump",
    "crossing": "Crossing",
    "give_way": "Give_Way",
    "junction": "Junction",
    "no_exit": "No_Exit",
    "railway": "Railway",
    "roundabout": "Roundabout",
    "station": "Station",
    "stop": "Stop",
    "traffic_calming": "Traffic_Calming",
    "traffic_signal": "Traffic_Signal",
    "turning_loop": "Turning_Loop",
}

CATEGORICAL_COLS = set(CATEGORICAL)


def build_row(item: AccidentInput) -> pd.DataFrame:
    data = item.model_dump()
    row = {}

    for field, col in COLUMN_MAP.items():
        row[col] = data.get(field)

    start, end = data.get("start_time"), data.get("end_time")
    row["Hour"] = start.hour if start else np.nan
    row["DayOfWeek"] = start.weekday() if start else np.nan
    row["Month"] = start.month if start else np.nan

    if start and end:
        minutes = (end - start).total_seconds() / 60.0
        row["Duration_Min"] = min(minutes, 1440) if minutes >= 0 else np.nan
    else:
        row["Duration_Min"] = np.nan

    df = pd.DataFrame([row])

    for col in features:
        if col not in df.columns:
            df[col] = np.nan

        if col in CATEGORICAL_COLS:
            df[col] = df[col].astype("object").where(df[col].notna(), "Unknown")
        else:
            df[col] = pd.to_numeric(df[col].astype("float64"), errors="coerce")

    return df[features]


FORM_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>US Accidents Severity Predictor</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:720px;margin:24px auto;padding:0 16px;color:#1a1a1a}
 h1{font-size:1.4rem} .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
 label{display:flex;flex-direction:column;font-size:.85rem;gap:4px}
 input,select{padding:8px;border:1px solid #bbb;border-radius:6px;font-size:.95rem}
 .checks{display:flex;gap:16px;flex-wrap:wrap;margin:14px 0}
 .checks label{flex-direction:row;align-items:center;gap:6px}
 button{background:#0b5fff;color:#fff;border:0;padding:12px 22px;border-radius:8px;font-size:1rem;cursor:pointer}
 button:disabled{opacity:.6} #out{margin-top:20px}
 .sev{font-size:1.6rem;font-weight:700;margin-bottom:10px}
 .bar{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:.9rem}
 .bar span.n{width:90px}.bar .t{flex:1;background:#e6e6e6;border-radius:4px;height:14px}
 .bar .f{background:#0b5fff;height:14px;border-radius:4px}
 .err{color:#b00020;white-space:pre-wrap} a{color:#0b5fff}
 @media(max-width:520px){.grid{grid-template-columns:1fr}}
</style></head><body>
<h1>US Accidents Severity Predictor</h1>
<p>Fill in what you know and press <b>Predict</b>. Empty fields are filled with typical values. <a href="/docs">API docs</a></p>
<form id="f">
 <div class="grid">
  <label>Start time<input type="datetime-local" name="start_time" value="2023-01-15T08:30"></label>
  <label>State (e.g. CA)<input name="state" value="CA" maxlength="2"></label>
  <label>Start latitude<input type="number" step="any" name="start_lat" value="34.05"></label>
  <label>Start longitude<input type="number" step="any" name="start_lng" value="-118.24"></label>
  <label>Temperature (F)<input type="number" step="any" name="temperature_f" value="52"></label>
  <label>Humidity (%)<input type="number" step="any" name="humidity_pct" value="85"></label>
  <label>Visibility (mi)<input type="number" step="any" name="visibility_mi" value="3"></label>
  <label>Wind speed (mph)<input type="number" step="any" name="wind_speed_mph" value="12"></label>
  <label>Precipitation (in)<input type="number" step="any" name="precipitation_in" value="0.1"></label>
  <label>Pressure (in)<input type="number" step="any" name="pressure_in" value="29.9"></label>
  <label>Weather condition
   <select name="weather_condition">
    <option>Fair</option><option>Cloudy</option><option>Mostly Cloudy</option>
    <option>Partly Cloudy</option><option selected>Light Rain</option><option>Rain</option>
    <option>Heavy Rain</option><option>Fog</option><option>Snow</option><option>Thunderstorm</option>
   </select></label>
  <label>Day or night
   <select name="sunrise_sunset"><option selected>Day</option><option>Night</option></select></label>
 </div>
 <div class="checks">
  <label><input type="checkbox" name="junction" checked> Junction</label>
  <label><input type="checkbox" name="traffic_signal" checked> Traffic signal</label>
  <label><input type="checkbox" name="crossing"> Crossing</label>
  <label><input type="checkbox" name="stop"> Stop sign</label>
  <label><input type="checkbox" name="railway"> Railway</label>
  <label><input type="checkbox" name="roundabout"> Roundabout</label>
 </div>
 <button id="b" type="submit">Predict</button>
</form>
<div id="out"></div>
<script>
const f=document.getElementById('f'),out=document.getElementById('out'),b=document.getElementById('b');
f.addEventListener('submit',async e=>{
 e.preventDefault(); b.disabled=true; out.textContent='Predicting...';
 const body={};
 for(const el of f.elements){
  if(!el.name) continue;
  if(el.type==='checkbox'){ if(el.checked) body[el.name]=true; continue; }
  if(el.value==='') continue;
  body[el.name]= el.type==='number' ? parseFloat(el.value) : el.value;
 }
 try{
  const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();
  if(!r.ok){ out.innerHTML='<div class="err"></div>'; out.firstChild.textContent='Error '+r.status+': '+JSON.stringify(d.detail,null,2); }
  else{
   let h='<div class="sev">Predicted severity: '+d.severity+'</div>';
   for(const [k,v] of Object.entries(d.probabilities)){
    h+='<div class="bar"><span class="n">'+k+'</span><div class="t"><div class="f" style="width:'+(v*100)+'%"></div></div><span>'+(v*100).toFixed(1)+'%</span></div>';
   }
   out.innerHTML=h;
  }
 }catch(err){ out.innerHTML='<div class="err"></div>'; out.firstChild.textContent='Request failed: '+err; }
 b.disabled=false;
});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/predict", response_class=HTMLResponse, include_in_schema=False)
def form_page():
    # Opening /predict in a browser (a GET request) now shows this form.
    return FORM_HTML


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_file": MODEL_PATH,
        "best_model_name": artifact.get("best_model_name"),
        "best_config_name": artifact.get("best_config_name"),
        "data_source": artifact.get("data_source"),
        "n_features": len(features),
    }


@app.post("/predict")
def predict(item: AccidentInput):
    try:
        X = processor.transform(build_row(item))
        proba = model.predict_proba(X)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    idx = int(np.argmax(proba))

    return {
        "severity": SEVERITY_LABELS[idx],
        "probabilities": {
            f"Severity {s}": round(float(p), 4)
            for s, p in zip(SEVERITY_LABELS, proba)
        },
    }
