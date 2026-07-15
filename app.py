
import json
import os
import re
from datetime import datetime, date
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
JST = ZoneInfo("Asia/Tokyo")

VENUES = {
    1:"桐生", 2:"戸田", 3:"江戸川", 4:"平和島", 5:"多摩川", 6:"浜名湖",
    7:"蒲郡", 8:"常滑", 9:"津", 10:"三国", 11:"びわこ", 12:"住之江",
    13:"尼崎", 14:"鳴門", 15:"丸亀", 16:"児島", 17:"宮島", 18:"徳山",
    19:"下関", 20:"若松", 21:"芦屋", 22:"福岡", 23:"唐津", 24:"大村"
}
VENUE_ALIASES = {v:k for k,v in VENUES.items()}
VENUE_ALIASES.update({"琵琶湖":11, "びわ湖":11, "からつ":23, "おおむら":24})

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15",
    "Accept-Language": "ja-JP,ja;q=0.9",
}

FIRST_MODEL_PATH = MODEL_DIR / "first_model.joblib"
SECOND_MODEL_PATH = MODEL_DIR / "second_conditional_model.joblib"
THIRD_MODEL_PATH = MODEL_DIR / "third_conditional_model.joblib"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"

_loaded = None

def load_assets():
    global _loaded
    if _loaded is not None:
        return _loaded

    required = [
        FIRST_MODEL_PATH,
        SECOND_MODEL_PATH,
        THIRD_MODEL_PATH,
        SCHEMA_PATH,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))

    first_model = joblib.load(FIRST_MODEL_PATH)
    second_model = joblib.load(SECOND_MODEL_PATH)
    third_model = joblib.load(THIRD_MODEL_PATH)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    _loaded = {
        "first_model": first_model,
        "second_model": second_model,
        "third_model": third_model,
        "schema": schema,
    }
    return _loaded

def parse_race_input(text: str):
    s = re.sub(r"\s+", "", str(text))
    m = re.fullmatch(r"(.+?)(\d{1,2})R?", s, flags=re.I)
    if not m:
        raise ValueError("race must look like 蒲郡12R")

    venue_name = m.group(1)
    race_no = int(m.group(2))

    if venue_name not in VENUE_ALIASES:
        raise ValueError(f"unknown venue: {venue_name}")
    if not 1 <= race_no <= 12:
        raise ValueError("race number must be 1-12")

    return VENUE_ALIASES[venue_name], race_no

def flatten_columns(df):
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(x) for x in col if str(x) != "nan").strip("_")
            for col in out.columns
        ]
    else:
        out.columns = [str(c) for c in out.columns]
    return out

def num(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else np.nan

def get_html(page, hd, venue_code, race_no):
    url = f"https://www.boatrace.jp/owpc/pc/race/{page}"
    params = {"hd": hd, "jcd": f"{venue_code:02d}", "rno": race_no}
    res = requests.get(url, params=params, headers=HEADERS, timeout=30)
    res.raise_for_status()
    res.encoding = res.apparent_encoding or "utf-8"
    return res.text, res.url

def find_main_racelist_table(html):
    tables = [flatten_columns(t) for t in pd.read_html(StringIO(html))]
    candidates = []
    for t in tables:
        text = " ".join(map(str, t.columns)) + " " + " ".join(
            t.astype(str).head(10).fillna("").values.ravel()
        )
        score = sum(k in text for k in ["登録番号", "全国", "当地", "モーター", "ボート"])
        if len(t) >= 6:
            candidates.append((score, len(t.columns), t))
    if not candidates:
        raise RuntimeError("racelist table not found")
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]

def parse_racelist(html, target_date, venue_code, race_no):
    t = find_main_racelist_table(html).iloc[:6].reset_index(drop=True)
    rows = []

    for i, row in t.iterrows():
        vals = [str(v).strip() for v in row.tolist()]
        joined = " | ".join(vals)

        reg_class = re.search(r"(\d{4})\s*/?\s*(A1|A2|B1|B2)", joined)
        if reg_class:
            registration = int(reg_class.group(1))
            klass = reg_class.group(2)
        else:
            rev = re.search(r"(A1|A2|B1|B2)\s*/?\s*(\d{4})", joined)
            if not rev:
                raise RuntimeError(f"registration parse failed lane={i+1}")
            klass = rev.group(1)
            registration = int(rev.group(2))

        name = f"{i+1}号艇"
        for v in vals:
            if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", v) and not any(
                k in v for k in ["全国","当地","モーター","ボート","成績","支部","出身"]
            ):
                cleaned = re.sub(r"\s+", "", v)
                if 2 <= len(cleaned) <= 12:
                    name = cleaned
                    break

        branch = "不明"
        for v in vals:
            m = re.search(r"([\u3400-\u9fff]{1,4})\s*/\s*([\u3400-\u9fff]{1,4})", v)
            if m:
                branch = m.group(1)
                break

        age_m = re.search(r"(\d{2})歳", joined)
        weight_m = re.search(r"(\d{2}(?:\.\d)?)kg", joined)

        def by_keywords(include):
            for c, v in row.items():
                cs = str(c)
                if all(k in cs for k in include):
                    value = num(v)
                    if not pd.isna(value):
                        return value
            return np.nan

        rows.append({
            "date": target_date.isoformat(),
            "venue_code": venue_code,
            "venue": VENUES[venue_code],
            "race_no": race_no,
            "race_title_program": "当日公式",
            "lane": i + 1,
            "registration": registration,
            "player_name_program": name,
            "age": int(age_m.group(1)) if age_m else np.nan,
            "branch": branch,
            "weight": float(weight_m.group(1)) if weight_m else np.nan,
            "class": klass,
            "national_win_rate": by_keywords(["全国", "勝率"]),
            "national_2rate": by_keywords(["全国", "2連率"]),
            "local_win_rate": by_keywords(["当地", "勝率"]),
            "local_2rate": by_keywords(["当地", "2連率"]),
            "motor_no_program": by_keywords(["モーター", "No"]),
            "motor_2rate": by_keywords(["モーター", "2連率"]),
            "boat_no_program": by_keywords(["ボート", "No"]),
            "boat_2rate": by_keywords(["ボート", "2連率"]),
        })

    race = pd.DataFrame(rows)
    if len(race) != 6:
        raise RuntimeError(f"expected 6 boats, got {len(race)}")
    return race

def find_col(df, keywords, exclude=()):
    for c in df.columns:
        name = re.sub(r"\s+", "", str(c))
        if all(k in name for k in keywords) and not any(x in name for x in exclude):
            return c
    return None

def numeric_series(series):
    return pd.to_numeric(
        series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )

def parse_beforeinfo(html):
    tables = [flatten_columns(t) for t in pd.read_html(StringIO(html))]
    scored = []
    for t in tables:
        text = " ".join(map(str, t.columns)) + " " + " ".join(
            t.astype(str).head(10).fillna("").values.ravel()
        )
        score = sum(k in text for k in ["展示", "チルト", "体重", "部品交換"])
        if len(t) >= 6:
            scored.append((score, len(t.columns), t))
    if not scored:
        raise RuntimeError("beforeinfo table not found")

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    b = scored[0][2].head(6).reset_index(drop=True)
    b.columns = [re.sub(r"\s+", "", str(c)) for c in b.columns]

    time_col = find_col(b, ["展示", "タイム"]) or find_col(b, ["展示"], exclude=("ST",))
    tilt_col = find_col(b, ["チルト"])
    weight_col = find_col(b, ["体重"])
    parts_col = find_col(b, ["部品"]) or find_col(b, ["交換"])

    out = pd.DataFrame({"lane": np.arange(1, 7)})
    out["exhibition_time"] = numeric_series(b[time_col]) if time_col else np.nan
    out["tilt"] = numeric_series(b[tilt_col]) if tilt_col else np.nan
    out["weight_before"] = numeric_series(b[weight_col]) if weight_col else np.nan
    out["parts_exchange"] = (
        b[parts_col].astype(str).replace("nan", "") if parts_col else ""
    )

    out["exhibition_st"] = np.nan
    for t in tables:
        t2 = t.copy()
        t2.columns = [re.sub(r"\s+", "", str(c)) for c in t2.columns]
        st_col = find_col(t2, ["ST"])
        if st_col is None:
            continue
        values = numeric_series(t2[st_col])
        if values.between(-0.30, 1.50).sum() >= 4:
            out["exhibition_st"] = values.head(6).reset_index(drop=True)
            break

    if out["exhibition_time"].notna().sum() < 4:
        for c in b.columns:
            values = numeric_series(b[c])
            if values.between(6.0, 8.0).sum() >= 4:
                out["exhibition_time"] = values
                break

    out["parts_exchange_flag"] = (
        out["parts_exchange"].fillna("").astype(str).str.strip().ne("").astype(int)
    )
    return out

def build_features(race, schema):
    feature_cols = schema["feature_cols"]
    categorical_cols = schema["categorical_cols"]

    race = race.copy()
    race["date"] = pd.to_datetime(race["date"])
    race["month"] = race["date"].dt.month
    race["dayofweek"] = race["date"].dt.dayofweek
    race["is_weekend"] = race["dayofweek"].isin([5, 6]).astype(int)

    numeric_defaults = {
        "registration":0, "age":0, "weight":0,
        "national_win_rate":0, "national_2rate":0,
        "local_win_rate":0, "local_2rate":0,
        "motor_no_program":0, "motor_2rate":0,
        "boat_no_program":0, "boat_2rate":0,
        "exhibition_time":0, "exhibition_st":0,
        "tilt":0, "weight_before":0, "parts_exchange_flag":0,
    }
    for c, default in numeric_defaults.items():
        if c not in race.columns:
            race[c] = default
        race[c] = pd.to_numeric(race[c], errors="coerce")
        med = race[c].median()
        if pd.isna(med):
            med = default
        race[c] = race[c].fillna(med)

    relative_cols = [
        "national_win_rate","national_2rate",
        "local_win_rate","local_2rate",
        "motor_2rate","boat_2rate",
        "exhibition_time","exhibition_st",
        "tilt","weight_before",
    ]
    for c in relative_cols:
        race[f"{c}_race_diff"] = race[c] - race[c].mean()
        ascending = c in ["exhibition_time","exhibition_st","weight_before"]
        race[f"{c}_race_rank"] = race[c].rank(method="average", ascending=ascending)

    race["lane_x_national_win"] = race["lane"] * race["national_win_rate"]
    race["lane_x_local_win"] = race["lane"] * race["local_win_rate"]
    race["lane_x_motor"] = race["lane"] * race["motor_2rate"]
    race["lane_x_exhibition"] = race["lane"] * race["exhibition_time"]

    for c in categorical_cols:
        if c not in race.columns:
            race[c] = "不明"
        race[c] = race[c].fillna("不明").astype("category")

    missing = [c for c in feature_cols if c not in race.columns]
    if missing:
        raise KeyError("missing features: " + ", ".join(missing))

    return race

def parse_odds_table(html):
    tables = pd.read_html(StringIO(html), header=None)
    best = pd.DataFrame()

    for table in tables:
        t = table.copy()
        if t.shape[0] < 15 or t.shape[1] < 18:
            continue

        for start in range(0, t.shape[1] - 18 + 1):
            part = t.iloc[:, start:start + 18].copy()
            parsed = []

            for first in range(1, 7):
                group = part.iloc[:, (first-1)*3:first*3].copy()
                group.columns = ["second","third","odds"]

                group["second"] = pd.to_numeric(group["second"], errors="coerce").ffill()
                group["third"] = pd.to_numeric(group["third"], errors="coerce")
                group["odds"] = pd.to_numeric(
                    group["odds"].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )

                for row in group.itertuples(index=False):
                    if pd.isna(row.second) or pd.isna(row.third) or pd.isna(row.odds):
                        continue
                    second, third, odd = int(row.second), int(row.third), float(row.odds)
                    if 1 <= second <= 6 and 1 <= third <= 6 and len({first,second,third}) == 3 and odd > 0:
                        parsed.append({"combo":f"{first}-{second}-{third}", "odds":odd})

            result = pd.DataFrame(parsed).drop_duplicates("combo")
            if len(result) > len(best):
                best = result

    if best.empty:
        raise RuntimeError("odds table parse failed")
    return best

def predict_internal(race_text: str, date_text: str | None = None):
    assets = load_assets()
    first_model = assets["first_model"]
    second_model = assets["second_model"]
    third_model = assets["third_model"]
    schema = assets["schema"]

    feature_cols = schema["feature_cols"]
    categorical_cols = schema["categorical_cols"]
    context_cols = schema["context_cols"]
    second_cols = schema["second_cols"]
    third_cols = schema["third_cols"]
    x_cols = feature_cols + categorical_cols

    venue_code, race_no = parse_race_input(race_text)
    target_date = date.fromisoformat(date_text) if date_text else datetime.now(JST).date()
    hd = target_date.strftime("%Y%m%d")

    racelist_html, _ = get_html("racelist", hd, venue_code, race_no)
    before_html, _ = get_html("beforeinfo", hd, venue_code, race_no)
    odds_html, _ = get_html("odds3t", hd, venue_code, race_no)

    race = parse_racelist(racelist_html, target_date, venue_code, race_no)
    before = parse_beforeinfo(before_html)
    race = race.merge(before, on="lane", how="left", validate="one_to_one")

    for c in ["exhibition_time","exhibition_st","tilt","weight_before"]:
        race[c] = pd.to_numeric(race[c], errors="coerce")
        med = race[c].median()
        race[c] = race[c].fillna(0 if pd.isna(med) else med)

    race = build_features(race, schema).sort_values("lane").copy()

    raw1 = first_model.predict_proba(race[x_cols])[:,1]
    race["p1"] = raw1 / raw1.sum()
    rows = []

    for first_lane in range(1,7):
        winner = race[race["lane"] == first_lane].iloc[0]
        sec = race[race["lane"] != first_lane].copy()

        for c in context_cols:
            sec[f"winner_{c}"] = winner[c]
            sec[f"candidate_minus_winner_{c}"] = sec[c] - winner[c]

        raw2 = second_model.predict_proba(sec[second_cols])[:,1]
        sec["p2"] = raw2 / raw2.sum()

        for _, second in sec.iterrows():
            second_lane = int(second["lane"])
            third = race[~race["lane"].isin([first_lane, second_lane])].copy()

            for c in context_cols:
                third[f"winner_{c}"] = winner[c]
                third[f"second_{c}"] = second[c]
                third[f"candidate_minus_winner_{c}"] = third[c] - winner[c]
                third[f"candidate_minus_second_{c}"] = third[c] - second[c]

            raw3 = third_model.predict_proba(third[third_cols])[:,1]
            third["p3"] = raw3 / raw3.sum()

            for _, r3 in third.iterrows():
                rows.append({
                    "combo": f"{first_lane}-{second_lane}-{int(r3['lane'])}",
                    "probability": float(winner["p1"]) * float(second["p2"]) * float(r3["p3"]),
                })

    pred = pd.DataFrame(rows)
    pred["probability"] /= pred["probability"].sum()

    odds = parse_odds_table(odds_html)
    out = pred.merge(odds, on="combo", how="left")

    # Conservative calibration
    temperature = 1.8
    p = out["probability"].clip(lower=1e-12).to_numpy()
    scaled = np.exp(np.log(p) / temperature)
    scaled /= scaled.sum()
    out["calibrated_probability"] = scaled

    out["market_raw_probability"] = 1 / out["odds"]
    out["market_probability"] = out["market_raw_probability"] / out["market_raw_probability"].sum()
    out["safe_probability"] = 0.75*out["calibrated_probability"] + 0.25*out["market_probability"]
    out["safe_probability"] /= out["safe_probability"].sum()
    out["safe_EV"] = out["safe_probability"] * out["odds"]
    out["market_disagreement"] = out["safe_probability"] / out["market_probability"]

    page_text = BeautifulSoup(racelist_html + before_html, "html.parser").get_text(" ", strip=True)
    special_keywords = ["安定板","1200m","進入固定","周回短縮","展示航走中止","レース中止"]
    specials = [x for x in special_keywords if x in page_text]

    rec = out[
        (out["safe_probability"] >= 0.008)
        & (out["safe_EV"] >= 1.15)
        & (out["odds"] <= 150)
        & (out["market_disagreement"] <= 2.5)
    ].copy()

    if specials:
        rec = rec[(rec["safe_EV"] >= 1.25) & (rec["odds"] <= 100)]

    rec = rec.sort_values(["safe_EV","safe_probability"], ascending=False).head(6)

    return {
        "race": race_text,
        "date": target_date.isoformat(),
        "venue_code": venue_code,
        "race_no": race_no,
        "special_conditions": specials,
        "fetched_at": datetime.now(JST).isoformat(),
        "recommendations": [
            {
                "combo": r.combo,
                "safe_probability": round(float(r.safe_probability), 6),
                "odds": round(float(r.odds), 1),
                "safe_EV": round(float(r.safe_EV), 3),
                "market_disagreement": round(float(r.market_disagreement), 3),
            }
            for r in rec.itertuples()
        ],
        "top10": [
            {
                "combo": r.combo,
                "probability": round(float(r.probability), 6),
            }
            for r in out.sort_values("probability", ascending=False).head(10).itertuples()
        ],
    }

@app.get("/")
def index():
    return jsonify({
        "service": "boatrace-ai",
        "status": "ok",
        "usage": "/predict?race=蒲郡12R"
    })

@app.get("/health")
def health():
    try:
        load_assets()
        return jsonify({"status":"ok","models_loaded":True})
    except Exception as e:
        return jsonify({"status":"error","error":str(e)}), 500

@app.get("/predict")
def predict():
    race_text = request.args.get("race", "").strip()
    date_text = request.args.get("date")

    if not race_text:
        return jsonify({"error":"race is required, e.g. 蒲郡12R"}), 400

    try:
        return jsonify(predict_internal(race_text, date_text))
    except Exception as e:
        return jsonify({"error":type(e).__name__, "message":str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
