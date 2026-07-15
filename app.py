
import json
import os
import re
from datetime import datetime, date, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo
import math

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from google.cloud import firestore
from flask import Flask, jsonify, request, render_template_string

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


def allocate_stakes_by_odds(
    recommendations,
    budget=1000,
    unit=100,
    target_payout=20000,
    min_stake=100,
):
    """
    候補4点以内:
      的中時払戻が2万円以上になる最小の100円単位で購入。
      それ以上は増額しない。
      1点上限は設けない。

    候補5点以上:
      全候補を残し、1/odds比例で合計1,000円を配分。
      各点最低100円。
    """
    if not recommendations:
        return []

    budget = int(budget)
    unit = int(unit)
    target_payout = int(target_payout)
    min_stake = int(min_stake)

    # 4候補以内は「2万円以上になる最小額」
    if len(recommendations) <= 4:
        allocated = []

        for rec in recommendations:
            odds = max(float(rec["odds"]), 1.01)

            # 2万円 ÷ オッズを100円単位で切り上げ
            stake_units = math.ceil(
                target_payout / odds / unit
            )
            stake = max(
                min_stake,
                int(stake_units * unit),
            )

            allocated.append({
                **rec,
                "stake": stake,
            })

        allocation_mode = "minimum_target_payout"

    # 5候補以上は合計1,000円を全候補へ配分
    else:
        max_picks = budget // unit
        recommendations = recommendations[:max_picks]
        count = len(recommendations)

        odds = np.array(
            [
                max(float(rec["odds"]), 1.01)
                for rec in recommendations
            ],
            dtype=float,
        )

        total_units = budget // unit
        units = np.ones(count, dtype=int)
        remaining_units = total_units - count

        weights = 1.0 / odds
        weights /= weights.sum()

        raw_extra = weights * remaining_units
        floor_extra = np.floor(raw_extra).astype(int)
        units += floor_extra

        leftover = remaining_units - int(floor_extra.sum())

        if leftover > 0:
            order = np.argsort(
                -(raw_extra - floor_extra)
            )

            for idx in order[:leftover]:
                units[idx] += 1

        allocated = [
            {
                **rec,
                "stake": int(stake_units * unit),
            }
            for rec, stake_units in zip(
                recommendations,
                units,
            )
        ]

        allocation_mode = "full_budget"

    total_stake = sum(
        int(item["stake"])
        for item in allocated
    )

    for item in allocated:
        odds = float(item["odds"])
        stake = int(item["stake"])
        gross_if_hit = odds * stake

        item["gross_if_hit"] = round(gross_if_hit)
        item["net_if_hit"] = round(
            gross_if_hit - total_stake
        )
        item["race_budget"] = budget
        item["total_stake"] = total_stake
        item["target_payout"] = target_payout
        item["allocation_mode"] = allocation_mode

    allocated.sort(
        key=lambda item: (
            float(item.get("safe_EV") or 0),
            float(item.get("safe_probability") or 0),
        ),
        reverse=True,
    )

    return allocated


def fetch_active_venue_codes(target_date):
    """公式の本日のレース一覧から開催中の場コードを取得する。"""
    hd = target_date.strftime("%Y%m%d")
    url = "https://www.boatrace.jp/owpc/pc/race/index"
    res = requests.get(
        url,
        params={"hd": hd},
        headers=HEADERS,
        timeout=30,
    )
    res.raise_for_status()
    res.encoding = res.apparent_encoding or "utf-8"

    soup = BeautifulSoup(res.text, "html.parser")
    codes = set()

    for link in soup.find_all("a", href=True):
        href = str(link.get("href", ""))

        if "racelist" not in href:
            continue

        match = re.search(r"(?:\?|&)jcd=(\d{1,2})", href)

        if match:
            code = int(match.group(1))

            if code in VENUES:
                codes.add(code)

    return sorted(codes)


def parse_deadline_times(html):
    """
    出走表ページ上部の「締切予定時刻」から
    1R〜12Rの締切時刻を取得する。
    """
    tables = pd.read_html(StringIO(html), header=None)

    for table in tables:
        for _, row in table.iterrows():
            values = [
                str(value).strip()
                for value in row.tolist()
                if not pd.isna(value)
            ]
            joined = " ".join(values)

            if "締切予定時刻" not in joined:
                continue

            times = re.findall(
                r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)",
                joined,
            )
            formatted = [f"{int(h):02d}:{m}" for h, m in times]

            if len(formatted) >= 1:
                return formatted[:12]

    # HTML構造変更時のフォールバック
    page_text = BeautifulSoup(
        html,
        "html.parser",
    ).get_text(" ", strip=True)

    match = re.search(
        r"締切予定時刻(.{0,500})",
        page_text,
    )

    if match:
        times = re.findall(
            r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)",
            match.group(1),
        )
        formatted = [f"{int(h):02d}:{m}" for h, m in times]

        if formatted:
            return formatted[:12]

    raise RuntimeError("締切予定時刻を取得できませんでした")


def fetch_venue_deadlines(target_date, venue_code):
    hd = target_date.strftime("%Y%m%d")
    html, _ = get_html(
        "racelist",
        hd,
        venue_code,
        1,
    )
    times = parse_deadline_times(html)

    deadlines = []

    for index, time_text in enumerate(times, start=1):
        hour, minute = map(int, time_text.split(":"))
        deadline = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            hour,
            minute,
            tzinfo=JST,
        )

        deadlines.append({
            "venue_code": venue_code,
            "venue": VENUES[venue_code],
            "race_no": index,
            "deadline": deadline,
        })

    return deadlines


def already_auto_predicted(target_date, venue_code, race_no):
    doc_id = (
        f"{target_date.isoformat()}_"
        f"{int(venue_code):02d}_"
        f"{int(race_no):02d}"
    )
    snap = (
        get_firestore()
        .collection("predictions")
        .document(doc_id)
        .get()
    )

    if not snap.exists:
        return False

    data = snap.to_dict() or {}
    return data.get("source") == "auto"


def run_auto_predictions(
    now=None,
    min_minutes_before=3,
    max_minutes_before=7,
    max_races=8,
):
    """
    Cloud Schedulerから5分おきに呼ぶ。
    締切3〜7分前の未予想レースのみ自動予想する。
    """
    now = now or datetime.now(JST)
    target_date = now.date()

    venue_codes = fetch_active_venue_codes(target_date)
    due_races = []
    schedule_errors = []

    for venue_code in venue_codes:
        try:
            deadlines = fetch_venue_deadlines(
                target_date,
                venue_code,
            )

            for race in deadlines:
                minutes_before = (
                    race["deadline"] - now
                ).total_seconds() / 60

                if (
                    min_minutes_before
                    <= minutes_before
                    <= max_minutes_before
                ):
                    due_races.append({
                        **race,
                        "minutes_before": round(
                            minutes_before,
                            1,
                        ),
                    })

        except Exception as exc:
            schedule_errors.append({
                "venue": VENUES.get(
                    venue_code,
                    str(venue_code),
                ),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })

    due_races.sort(key=lambda item: item["deadline"])
    due_races = due_races[:max_races]

    predicted = []
    skipped = []
    prediction_errors = []

    for race in due_races:
        venue_code = race["venue_code"]
        race_no = race["race_no"]
        race_text = f"{VENUES[venue_code]}{race_no}R"

        if already_auto_predicted(
            target_date,
            venue_code,
            race_no,
        ):
            skipped.append({
                "race": race_text,
                "reason": "already_predicted",
            })
            continue

        try:
            result = predict_internal(
                race_text,
                target_date.isoformat(),
            )
            log_id = save_prediction_log(
                result,
                source="auto",
            )

            predicted.append({
                "race": race_text,
                "deadline": (
                    race["deadline"].isoformat()
                ),
                "minutes_before": (
                    race["minutes_before"]
                ),
                "decision": (
                    "推奨"
                    if result.get("recommendations")
                    else "見送り"
                ),
                "recommendations": len(
                    result.get(
                        "recommendations",
                        [],
                    )
                ),
                "log_id": log_id,
            })

        except Exception as exc:
            prediction_errors.append({
                "race": race_text,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })

    return {
        "status": "ok",
        "checked_at": now.isoformat(),
        "active_venues": len(venue_codes),
        "due_races": len(due_races),
        "predicted": predicted,
        "skipped": skipped,
        "schedule_errors": schedule_errors[:10],
        "prediction_errors": prediction_errors[:10],
    }



def detect_race_grade(html):
    """
    公式ページの本文・HTML属性・大会名から
    SG/G1/G2/G3/一般を判定する。
    """
    soup = BeautifulSoup(html, "html.parser")

    page_text = soup.get_text(
        " ",
        strip=True,
    ).upper()

    # imgのalt/src、class名などにグレード表記がある場合も拾う
    html_signals = " ".join([
        str(html),
        " ".join(
            str(img.get("alt", ""))
            + " "
            + str(img.get("src", ""))
            for img in soup.find_all("img")
        ),
    ]).upper()

    compact = re.sub(
        r"\s+",
        "",
        page_text + " " + html_signals,
    )

    # SGの代表的大会名
    sg_keywords = [
        "ボートレースクラシック",
        "ボートレースオールスター",
        "グランドチャンピオン",
        "オーシャンカップ",
        "ボートレースメモリアル",
        "ボートレースダービー",
        "チャレンジカップ",
        "グランプリ",
    ]

    # G2の代表的大会名
    g2_keywords = [
        "モーターボート大賞",
        "レディースオールスター",
        "全国ボートレース甲子園",
        "モーターボート誕生祭",
        "秩父宮妃記念杯",
    ]

    if (
        re.search(
            r"(^|[^A-Z0-9])SG([^A-Z0-9]|$)",
            page_text,
        )
        or "GRADE_SG" in html_signals
        or "ICON_SG" in html_signals
        or any(
            keyword.upper() in page_text
            for keyword in sg_keywords
        )
    ):
        return "SG"

    if (
        "GⅠ" in page_text
        or "ＧⅠ" in page_text
        or re.search(
            r"(^|[^A-Z0-9])G1([^A-Z0-9]|$)",
            page_text,
        )
        or "GRADE_G1" in html_signals
        or "ICON_G1" in html_signals
        or "/G1" in html_signals
    ):
        return "G1"

    if (
        "GⅡ" in page_text
        or "ＧⅡ" in page_text
        or re.search(
            r"(^|[^A-Z0-9])G2([^A-Z0-9]|$)",
            page_text,
        )
        or "GRADE_G2" in html_signals
        or "ICON_G2" in html_signals
        or "/G2" in html_signals
        or any(
            keyword.upper() in page_text
            for keyword in g2_keywords
        )
    ):
        return "G2"

    if (
        "GⅢ" in page_text
        or "ＧⅢ" in page_text
        or re.search(
            r"(^|[^A-Z0-9])G3([^A-Z0-9]|$)",
            page_text,
        )
        or "GRADE_G3" in html_signals
        or "ICON_G3" in html_signals
        or "/G3" in html_signals
    ):
        return "G3"

    return "一般"


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

    combined_race_html = racelist_html + before_html
    race_grade = detect_race_grade(combined_race_html)

    page_text = BeautifulSoup(
        combined_race_html,
        "html.parser",
    ).get_text(" ", strip=True)
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

    rec = rec.sort_values(
        ["safe_EV", "safe_probability"],
        ascending=False
    ).head(6)

    recommendation_rows = [
        {
            "combo": r.combo,
            "safe_probability": round(float(r.safe_probability), 6),
            "odds": round(float(r.odds), 1),
            "safe_EV": round(float(r.safe_EV), 3),
            "market_disagreement": round(float(r.market_disagreement), 3),
        }
        for r in rec.itertuples()
    ]

    recommendation_rows = allocate_stakes_by_odds(
        recommendation_rows,
        budget=1000,
        unit=100,
        target_payout=20000,
        min_stake=100,
    )

    canonical_race_name = f"{VENUES[venue_code]}{race_no}R"

    return {
        "race": canonical_race_name,
        "date": target_date.isoformat(),
        "venue_code": venue_code,
        "race_no": race_no,
        "race_grade": race_grade,
        "special_conditions": specials,
        "fetched_at": datetime.now(JST).isoformat(),
        "budget": 1000,
        "race_hit_probability": round(
            min(
                1.0,
                sum(
                    float(r.get("safe_probability") or 0)
                    for r in recommendation_rows
                ),
            ),
            6,
        ),
        "recommendations": recommendation_rows,
        "all_combo_predictions": [
            {
                "combo": r.combo,
                "safe_probability": round(
                    float(r.safe_probability),
                    6,
                ),
                "odds": round(float(r.odds), 1),
                "safe_EV": round(float(r.safe_EV), 3),
                "rank": int(rank),
            }
            for rank, r in enumerate(
                out.sort_values(
                    ["safe_probability", "safe_EV"],
                    ascending=False,
                ).itertuples(),
                start=1,
            )
        ],
        "top10": [
            {
                "combo": r.combo,
                "probability": round(float(r.probability), 6),
            }
            for r in out.sort_values("probability", ascending=False).head(10).itertuples()
        ],
    }


_firestore_client = None

def get_firestore():
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore.Client()
    return _firestore_client

def prediction_doc_id(result):
    # 同じ日・同じ会場・同じRは同じIDにして、
    # 後から出した予想で上書きする。
    return (
        f"{result['date']}_"
        f"{int(result['venue_code']):02d}_"
        f"{int(result['race_no']):02d}"
    )

def save_prediction_log(result, source="manual"):
    db = get_firestore()
    recommendations = result.get("recommendations", [])
    now = datetime.now(JST)

    payload = {
        "source": source,
        "race": result["race"],
        "date": result["date"],
        "venue_code": int(result["venue_code"]),
        "race_no": int(result["race_no"]),
        "race_grade": result.get("race_grade", "一般"),
        "grade_detection_version": 2,
        "special_conditions": result.get("special_conditions", []),
        "fetched_at": result.get("fetched_at"),
        "created_at": now,
        "decision": "推奨" if recommendations else "見送り",
        "race_hit_probability": result.get(
            "race_hit_probability",
            0,
        ),
        "recommendations": recommendations,
        "all_combo_predictions": result.get(
            "all_combo_predictions",
            [],
        ),
        "result_checked": False,
        "result_combo": None,
        "trifecta_payout": None,
        "stake": sum(
            int(r.get("stake") or 0)
            for r in recommendations
        ),
        "payout": None,
        "profit": None,
        "hit": None,
    }

    doc_id = prediction_doc_id(result)
    db.collection("predictions").document(doc_id).set(payload)
    return doc_id

def parse_result_combo(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    # 払戻表などにある三連単表記を優先
    for tr in soup.find_all("tr"):
        row_text = " ".join(tr.stripped_strings)
        if "3連単" not in row_text and "三連単" not in row_text:
            continue
        m = re.search(
            r"([1-6])\s*[-－]\s*([1-6])\s*[-－]\s*([1-6])",
            row_text
        )
        if m and len(set(m.groups())) == 3:
            return "-".join(m.groups())

    # ページ全体のフォールバック
    combos = re.findall(
        r"([1-6])\s*[-－]\s*([1-6])\s*[-－]\s*([1-6])",
        text
    )
    for combo in combos:
        if len(set(combo)) == 3:
            return "-".join(combo)

    raise RuntimeError("result not published yet")

def parse_trifecta_payout(html, combo):
    """
    公式結果ページの払戻表から、
    3連単の100円あたり払戻金だけを厳密に取得する。
    """
    target_combo = (
        str(combo)
        .replace("－", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace(" ", "")
    )

    def normalize_combo(value):
        return (
            str(value)
            .replace("－", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace(" ", "")
        )

    def extract_money(value):
        value = str(value).strip()

        # 通貨記号または「円」が付いたセルだけを金額として採用する。
        match = re.search(
            r"(?:[¥￥]\s*([\d,]+)|([\d,]+)\s*円)",
            value,
        )

        if not match:
            return None

        raw = match.group(1) or match.group(2)
        amount = int(raw.replace(",", ""))

        return amount if amount >= 100 else None

    soup = BeautifulSoup(html, "html.parser")

    # HTML表の各行をセル単位で確認する。
    for tr in soup.find_all("tr"):
        cells = [
            cell.get_text(" ", strip=True)
            for cell in tr.find_all(["th", "td"])
        ]

        if not cells:
            continue

        row_text = " ".join(cells)

        if "3連単" not in row_text and "三連単" not in row_text:
            continue

        combo_found = any(
            normalize_combo(cell) == target_combo
            for cell in cells
        )

        if not combo_found:
            continue

        for cell in cells:
            amount = extract_money(cell)

            if amount is not None:
                return amount

    # pandasで読み取れる表もセル単位で確認する。
    try:
        tables = pd.read_html(StringIO(html), header=None)

        for table in tables:
            for _, row in table.iterrows():
                cells = [
                    str(value).strip()
                    for value in row.tolist()
                    if not pd.isna(value)
                ]
                row_text = " ".join(cells)

                if (
                    "3連単" not in row_text
                    and "三連単" not in row_text
                ):
                    continue

                combo_found = any(
                    normalize_combo(cell) == target_combo
                    for cell in cells
                )

                if not combo_found:
                    continue

                for cell in cells:
                    amount = extract_money(cell)

                    if amount is not None:
                        return amount
    except Exception:
        pass

    return None

def fetch_official_result(date_text, venue_code, race_no):
    hd = str(date_text).replace("-", "")
    html, url = get_html(
        "raceresult",
        hd,
        int(venue_code),
        int(race_no),
    )
    combo = parse_result_combo(html)
    payout = parse_trifecta_payout(html, combo)
    grade = detect_race_grade(html)
    return combo, payout, url, grade

def update_pending_results(limit=100):
    db = get_firestore()
    # 未確認に加え、「的中なのに払戻0円」の過去データも再取得する。
    docs = (
        db.collection("predictions")
        .limit(max(limit, 200))
        .stream()
    )

    checked = 0
    updated = 0
    pending = 0
    errors = []

    for doc in docs:
        data = doc.to_dict()

        needs_check = data.get("result_checked") is not True

        # 過去に誤った払戻を保存した可能性があるため、
        # 的中済みレースはすべて公式結果から再計算する。
        needs_payout_recheck = data.get("hit") is True
        needs_grade_backfill = (
            int(
                data.get("grade_detection_version")
                or 0
            ) < 2
        )

        if (
            not needs_check
            and not needs_payout_recheck
            and not needs_grade_backfill
        ):
            continue

        checked += 1

        try:
            combo, trifecta_payout, _, race_grade = fetch_official_result(
                data["date"],
                data["venue_code"],
                data["race_no"]
            )

            recommendations = data.get("recommendations", [])
            combos = [str(x.get("combo", "")) for x in recommendations]
            hit = combo in combos

            all_combo_predictions = data.get(
                "all_combo_predictions",
                [],
            )

            result_prediction = next(
                (
                    item
                    for item in all_combo_predictions
                    if str(item.get("combo", "")) == combo
                ),
                None,
            )

            # 過去データは推奨買い目しか保存されていない場合がある
            if result_prediction is None:
                result_prediction = next(
                    (
                        item
                        for item in recommendations
                        if str(item.get("combo", "")) == combo
                    ),
                    None,
                )

            result_predicted_probability = None
            result_predicted_odds = None
            result_predicted_ev = None
            result_predicted_rank = None

            if result_prediction:
                result_predicted_probability = (
                    result_prediction.get("safe_probability")
                )
                result_predicted_odds = result_prediction.get("odds")
                result_predicted_ev = result_prediction.get("safe_EV")
                result_predicted_rank = result_prediction.get("rank")

            stake = int(
                data.get("stake")
                or sum(
                    int(x.get("stake") or 0)
                    for x in recommendations
                )
            )

            winning_stake = 0

            for item in recommendations:
                if str(item.get("combo", "")) == combo:
                    winning_stake = int(item.get("stake") or 0)
                    break

            payout = 0

            if (
                hit
                and trifecta_payout is not None
                and winning_stake > 0
            ):
                # 公式払戻は100円あたり
                payout = int(
                    round(
                        float(trifecta_payout)
                        * winning_stake
                        / 100
                    )
                )

            if hit and trifecta_payout is None:
                pending += 1
                continue

            profit = payout - stake

            doc.reference.update({
                "result_checked": True,
                "result_checked_at": datetime.now(JST),
                "result_combo": combo,
                "race_grade": race_grade,
                "grade_detection_version": 2,
                "result_predicted_probability": (
                    result_predicted_probability
                ),
                "result_predicted_odds": result_predicted_odds,
                "result_predicted_ev": result_predicted_ev,
                "result_predicted_rank": result_predicted_rank,
                "result_was_recommended": hit,
                "trifecta_payout": trifecta_payout,
                "hit": hit,
                "payout": payout,
                "profit": profit,
            })
            updated += 1

        except RuntimeError:
            pending += 1
        except Exception as e:
            errors.append({
                "doc_id": doc.id,
                "error": f"{type(e).__name__}: {e}"
            })

    return {
        "checked": checked,
        "updated": updated,
        "pending": pending,
        "errors": errors[:10],
    }


def delete_all_predictions():
    db = get_firestore()
    deleted = 0

    while True:
        docs = list(
            db.collection("predictions")
            .limit(200)
            .stream()
        )

        if not docs:
            break

        batch = db.batch()

        for doc in docs:
            batch.delete(doc.reference)
            deleted += 1

        batch.commit()

    return deleted


def build_stats():
    db = get_firestore()
    docs = list(
        db.collection("predictions")
        .order_by(
            "created_at",
            direction=firestore.Query.DESCENDING,
        )
        .limit(1000)
        .stream()
    )

    raw_rows = [
        doc.to_dict() | {"id": doc.id}
        for doc in docs
    ]

    latest_by_race = {}

    for row in raw_rows:
        key = (
            str(row.get("date")),
            int(row.get("venue_code") or 0),
            int(row.get("race_no") or 0),
        )
        current = latest_by_race.get(key)

        if current is None:
            latest_by_race[key] = row
            continue

        row_time = row.get("created_at")
        current_time = current.get("created_at")

        if current_time is None or (
            row_time is not None
            and row_time > current_time
        ):
            latest_by_race[key] = row

    rows = sorted(
        latest_by_race.values(),
        key=lambda r: (
            r.get("created_at")
            or datetime.min.replace(tzinfo=JST)
        ),
        reverse=True,
    )

    def normalized_grade(row):
        grade = str(
            row.get("race_grade")
            or "一般"
        ).upper()

        if grade == "SG":
            return "SG"
        if grade in {"G1", "GⅠ", "ＧⅠ"}:
            return "G1"
        if grade in {"G2", "GⅡ", "ＧⅡ"}:
            return "G2"
        if grade in {"G3", "GⅢ", "ＧⅢ"}:
            return "G3"

        return "一般"

    def summarize(target_rows):
        finished = [
            r for r in target_rows
            if r.get("result_checked") is True
        ]
        recommended = [
            r for r in finished
            if r.get("decision") == "推奨"
        ]

        total_stake = sum(
            int(r.get("stake") or 0)
            for r in recommended
        )
        total_payout = sum(
            int(r.get("payout") or 0)
            for r in recommended
        )
        total_profit = sum(
            int(r.get("profit") or 0)
            for r in recommended
        )
        hit_races = sum(
            1 for r in recommended
            if r.get("hit") is True
        )

        venue_map = {}

        for row in recommended:
            venue_code = row.get("venue_code")

            try:
                venue_code = int(venue_code)
            except (TypeError, ValueError):
                venue_code = None

            if venue_code in VENUES:
                venue = VENUES[venue_code]
            else:
                race_name = str(row.get("race") or "")
                venue = re.sub(
                    r"\d{1,2}R?$",
                    "",
                    race_name,
                    flags=re.I,
                ) or "不明"

            item = venue_map.setdefault(
                venue,
                {
                    "venue": venue,
                    "predictions": 0,
                    "hits": 0,
                    "stake": 0,
                    "payout": 0,
                    "probability_sum": 0.0,
                    "probability_count": 0,
                },
            )

            item["predictions"] += 1
            item["hits"] += (
                1 if row.get("hit") is True else 0
            )
            item["stake"] += int(
                row.get("stake") or 0
            )
            item["payout"] += int(
                row.get("payout") or 0
            )

            probability = row.get(
                "race_hit_probability"
            )

            if probability is None:
                probability = min(
                    1.0,
                    sum(
                        float(
                            x.get("safe_probability")
                            or 0
                        )
                        for x in row.get(
                            "recommendations",
                            [],
                        )
                    ),
                )

            item["probability_sum"] += float(
                probability or 0
            )
            item["probability_count"] += 1

        venue_stats = []

        for item in venue_map.values():
            count = item["predictions"]
            stake = item["stake"]
            payout = item["payout"]
            probability_count = item[
                "probability_count"
            ]

            venue_stats.append({
                "venue": item["venue"],
                "predictions": count,
                "hits": item["hits"],
                "stake": stake,
                "payout": payout,
                "hit_rate": (
                    item["hits"] / count
                    if count else None
                ),
                "average_race_hit_probability": (
                    item["probability_sum"]
                    / probability_count
                    if probability_count
                    else None
                ),
                "profit": payout - stake,
                "roi": (
                    payout / stake
                    if stake else None
                ),
            })

        venue_stats.sort(
            key=lambda x: (
                x["predictions"],
                x["hit_rate"] or 0,
            ),
            reverse=True,
        )

        recent = []

        for r in target_rows[:30]:
            recent.append({
                "id": r.get("id"),
                "source": r.get(
                    "source",
                    "manual",
                ),
                "race": r.get("race"),
                "race_grade": normalized_grade(r),
                "date": r.get("date"),
                "fetched_at": r.get("fetched_at"),
                "decision": r.get("decision"),
                "race_hit_probability": r.get(
                    "race_hit_probability",
                    min(
                        1.0,
                        sum(
                            float(
                                x.get(
                                    "safe_probability"
                                ) or 0
                            )
                            for x in r.get(
                                "recommendations",
                                [],
                            )
                        ),
                    ),
                ),
                "recommendations": r.get(
                    "recommendations",
                    [],
                ),
                "special_conditions": r.get(
                    "special_conditions",
                    [],
                ),
                "result_combo": r.get(
                    "result_combo"
                ),
                "result_predicted_probability": (
                    r.get(
                        "result_predicted_probability"
                    )
                ),
                "result_predicted_odds": r.get(
                    "result_predicted_odds"
                ),
                "result_predicted_ev": r.get(
                    "result_predicted_ev"
                ),
                "result_predicted_rank": r.get(
                    "result_predicted_rank"
                ),
                "result_was_recommended": r.get(
                    "result_was_recommended",
                    r.get("hit"),
                ),
                "hit": r.get("hit"),
                "stake": r.get("stake"),
                "payout": r.get("payout"),
                "profit": r.get("profit"),
            })

        return {
            "saved_predictions": len(target_rows),
            "finished_races": len(finished),
            "recommended_finished": len(
                recommended
            ),
            "hit_races": hit_races,
            "hit_rate": (
                hit_races / len(recommended)
                if recommended else None
            ),
            "total_stake": total_stake,
            "total_payout": total_payout,
            "total_profit": total_profit,
            "roi": (
                total_payout / total_stake
                if total_stake else None
            ),
            "venue_stats": venue_stats,
            "recent": recent,
        }

    groups = {
        "all": rows,
        "SG": [
            r for r in rows
            if normalized_grade(r) == "SG"
        ],
        "G1": [
            r for r in rows
            if normalized_grade(r) == "G1"
        ],
        "G2": [
            r for r in rows
            if normalized_grade(r) == "G2"
        ],
        "G3": [
            r for r in rows
            if normalized_grade(r) == "G3"
        ],
        "一般": [
            r for r in rows
            if normalized_grade(r) == "一般"
        ],
    }

    return {
        "groups": {
            key: summarize(group_rows)
            for key, group_rows in groups.items()
        }
    }

HTML_PAGE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>競艇AI</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0e1116;
      --panel: #171b22;
      --line: #2a303a;
      --text: #f4f6f8;
      --muted: #a9b0bb;
      --accent: #5ca7ff;
      --good: #53d18b;
      --warn: #ffca5c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                   "Noto Sans JP", sans-serif;
    }
    main {
      width: min(720px, 100%);
      margin: 0 auto;
      padding: 22px 16px 60px;
    }
    h1 { margin: 4px 0 6px; font-size: 28px; }
    .sub { color: var(--muted); margin-bottom: 22px; }
    form { display: flex; gap: 10px; margin-bottom: 18px; }
    input {
      min-width: 0;
      flex: 1;
      font-size: 18px;
      padding: 14px 15px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 0 18px;
      font-size: 16px;
      font-weight: 700;
      color: white;
      background: var(--accent);
    }
    .status {
      color: var(--muted);
      min-height: 24px;
      margin: 8px 0 16px;
    }
    .meta, .card, .empty {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 15px;
      margin-bottom: 12px;
    }
    .meta { line-height: 1.7; }
    .special { color: var(--warn); font-weight: 700; }
    .combo { font-size: 25px; font-weight: 800; margin-bottom: 8px; }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 7px 14px;
      color: var(--muted);
    }
    .grid b { color: var(--text); }
    .ev { color: var(--good) !important; }
    .empty { text-align: center; font-size: 20px; padding: 30px 15px; }
    .secondary { background: #343b46; }
    .danger { background: #c94a4a; }
    .stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .stat { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; }
    .stat small { display:block; color:var(--muted); margin-bottom:5px; }
    .stat strong { font-size:22px; }
    .history { margin-top:14px; }
    .history-row {
      border-bottom:1px solid var(--line);
      padding:12px 0;
      font-size:14px;
      cursor:pointer;
    }
    .history-row:last-child { border-bottom:0; }
    .history-summary {
      display:flex;
      justify-content:space-between;
      gap:10px;
      align-items:center;
    }
    .history-arrow { color:var(--muted); font-size:18px; }
    .history-detail {
      display:none;
      margin-top:10px;
      padding:12px;
      border-radius:10px;
      background:#10141a;
      border:1px solid var(--line);
    }
    .history-row.open .history-detail { display:block; }
    .history-pick {
      display:grid;
      grid-template-columns:1fr auto auto;
      gap:8px;
      padding:7px 0;
      border-bottom:1px solid var(--line);
    }
    .history-pick:last-child { border-bottom:0; }
    .history-muted { color:var(--muted); font-size:12px; }
    .foot {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      margin-top: 18px;
    }
  
    /* Mobile layout */
    @media (max-width: 640px) {
      .form-row,
      form {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 10px !important;
        align-items: stretch !important;
      }

      input[type="text"],
      input[type="search"],
      #raceInput {
        grid-column: 1 / -1 !important;
        width: 100% !important;
        min-width: 0 !important;
        height: 56px !important;
        box-sizing: border-box !important;
        font-size: 16px !important;
      }

      button {
        width: 100% !important;
        min-width: 0 !important;
        min-height: 56px !important;
        padding: 10px 8px !important;
        font-size: 16px !important;
        line-height: 1.25 !important;
        white-space: nowrap !important;
      }

      #refreshBtn {
        font-size: 15px !important;
      }

      .container,
      main {
        width: 100% !important;
        max-width: 100% !important;
        padding-left: 16px !important;
        padding-right: 16px !important;
        box-sizing: border-box !important;
      }

      .stats-grid {
        gap: 10px !important;
      }

      .card,
      .stat-card {
        min-width: 0 !important;
      }
    }

    @media (max-width: 390px) {
      button {
        font-size: 15px !important;
        padding-left: 6px !important;
        padding-right: 6px !important;
      }

      #refreshBtn {
        font-size: 14px !important;
      }
    }


    .grade-tabs {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 7px;
      margin: 14px 0;
    }
    .grade-tab {
      min-height: 44px;
      padding: 8px 4px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--muted);
      border-radius: 11px;
      font-size: 14px;
      white-space: nowrap;
    }
    .grade-tab.active {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    @media (max-width: 390px) {
      .grade-tabs { gap: 5px; }
      .grade-tab {
        font-size: 12px !important;
        padding: 7px 2px !important;
      }
    }

</style>
</head>
<body>
<main>
  <h1>🚤 競艇AI</h1>
  <div class="sub">レース名を入れるだけで保守的候補を表示</div>

  <form id="form">
    <input id="race" value="蒲郡12R" placeholder="例：浜名湖6R">
    <button type="submit">予想</button>
    <button type="button" id="statsBtn" class="secondary">成績</button>
    <button type="button" id="refreshBtn" class="secondary">結果を更新</button>
    <button type="button" id="deleteBtn" class="danger">全削除</button>
  </form>

  <div id="status" class="status"></div>
  <div id="result"></div>

  <div class="foot">
    生AI確率ではなく、補正確率・保守EV・市場乖離を使って候補を絞っています。<br>
    候補がない場合は「見送り」が正常です。
  </div>
</main>

<script>
const form = document.getElementById("form");
const input = document.getElementById("race");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");
const statsBtn = document.getElementById("statsBtn");
const refreshBtn = document.getElementById("refreshBtn");
const deleteBtn = document.getElementById("deleteBtn");

function esc(v) {
  return String(v ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const race = input.value.trim();
  if (!race) return;

  statusBox.textContent = "予想中… 初回は少し時間がかかります";
  resultBox.innerHTML = "";

  try {
    const res = await fetch("/predict?race=" + encodeURIComponent(race));
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.message || data.error || "予想に失敗しました");
    }

    const specials = data.special_conditions || [];
    let html = `
      <div class="meta">
        <b>${esc(data.race)}</b><br>
        日付：${esc(data.date)}<br>
        取得：${esc(data.fetched_at)}<br>
        投資上限：${Number(data.budget || 0).toLocaleString()}円<br>
        実際の投資額：${(data.recommendations || []).reduce(
          (sum, r) => sum + Number(r.stake || 0),
          0
        ).toLocaleString()}円<br>
        配分方式：${
          (data.recommendations || []).length <= 4
            ? "払戻20,000円以上になる最小額"
            : "全候補へ合計1,000円を配分"
        }<br>
        レース的中見込み：
        <b>${(Number(data.race_hit_probability || 0) * 100).toFixed(1)}%</b>
        ${
          specials.length
          ? `<br><span class="special">⚠️ ${esc(specials.join(" / "))}</span>`
          : ""
        }
      </div>
    `;

    const recs = data.recommendations || [];

    if (recs.length === 0) {
      html += `<div class="empty">今回は見送り</div>`;
    } else {
      recs.forEach((r, i) => {
        html += `
          <div class="card">
            <div class="combo">${i + 1}位　${esc(r.combo)}</div>
            <div class="grid">
              <span>補正確率</span>
              <b>${(Number(r.safe_probability) * 100).toFixed(2)}%</b>
              <span>現在オッズ</span>
              <b>${Number(r.odds).toFixed(1)}倍</b>
              <span>保守EV</span>
              <b class="ev">${Number(r.safe_EV).toFixed(2)}</b>
              <span>市場乖離</span>
              <b>${Number(r.market_disagreement).toFixed(2)}倍</b>
              <span>購入額</span>
              <b>${Number(r.stake || 0).toLocaleString()}円</b>
              <span>的中時払戻</span>
              <b>${Number(r.gross_if_hit || 0).toLocaleString()}円</b>
              <span>的中時収支</span>
              <b class="ev">${Number(r.net_if_hit || 0).toLocaleString()}円</b>
            </div>
          </div>
        `;
      });
    }

    resultBox.innerHTML = html;
    statusBox.textContent = data.log_id ? "予想完了・自動保存済み" : "予想完了";
  } catch (err) {
    statusBox.textContent = "エラー";
    resultBox.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
  }
});

statsBtn.addEventListener("click", async () => {
  statusBox.textContent = "成績を読み込み中…";
  resultBox.innerHTML = "";

  try {
    const res = await fetch("/stats");
    const payload = await res.json();

    if (!res.ok) {
      throw new Error(
        payload.message
        || payload.error
        || "成績取得失敗"
      );
    }

    const groups = payload.groups || {};
    const all = groups.all || {};

    const pct = (v) =>
      v == null
        ? "-"
        : (Number(v) * 100).toFixed(1) + "%";

    const yen = (v) =>
      Number(v || 0).toLocaleString() + "円";

    const labels = {
      SG: "SG",
      G1: "G1",
      G2: "G2",
      G3: "G3",
      "一般": "一般"
    };

    const renderGradeSection = (grade) => {
      const s = groups[grade] || {
        venue_stats: [],
        recent: []
      };

      const venues = s.venue_stats || [];

      const venueHtml = venues.length === 0
        ? `<div class="empty">
            確定済みの推奨予想がまだありません
          </div>`
        : venues.map((v) => `
            <div class="history-row">
              <div class="history-summary">
                <div>
                  <b>${esc(v.venue)}</b><br>
                  ${Number(v.hits || 0)}
                  / ${Number(v.predictions || 0)}的中
                  ・的中率 ${pct(v.hit_rate)}<br>
                  平均レース的中見込み
                  ${pct(v.average_race_hit_probability)}
                </div>
                <div style="text-align:right">
                  回収率 ${pct(v.roi)}<br>
                  ${yen(v.profit)}
                </div>
              </div>
            </div>
          `).join("");

      let recentHtml = "";

      (s.recent || []).forEach((r, index) => {
        const hit = r.hit === true
          ? "🎯 的中"
          : (
              r.hit === false
                ? "はずれ"
                : "未確定"
            );

        const recs = r.recommendations || [];
        let picksHtml = "";

        if (recs.length === 0) {
          picksHtml = `
            <div class="history-muted">
              この予想は見送りでした
            </div>
          `;
        } else {
          recs.forEach((pick, i) => {
            picksHtml += `
              <div class="history-pick">
                <b>${i + 1}位 ${esc(pick.combo)}</b>
                <span>
                  ${Number(pick.odds || 0).toFixed(1)}倍
                </span>
                <span>
                  ${Number(pick.stake || 0).toLocaleString()}円
                </span>
              </div>
              <div class="history-muted">
                補正確率
                ${(Number(pick.safe_probability || 0) * 100).toFixed(2)}%
                / 保守EV
                ${Number(pick.safe_EV || 0).toFixed(2)}
              </div>
            `;
          });
        }

        const conditions =
          (r.special_conditions || []).length
            ? `<div class="special">
                ⚠️ ${esc(r.special_conditions.join(" / "))}
              </div>`
            : "";

        recentHtml += `
          <div
            class="history-row"
            data-history-index="${index}"
          >
            <div class="history-summary">
              <div>
                <b>
                  ${esc(r.date)} ${esc(r.race)}
                  ${r.source === "auto" ? " 🤖" : ""}
                </b><br>
                ${esc(r.decision)} / ${hit}
                ${
                  r.profit == null
                    ? ""
                    : ` / ${yen(r.profit)}`
                }
              </div>
              <span class="history-arrow">›</span>
            </div>

            <div class="history-detail">
              <div class="history-muted">
                グレード：${esc(r.race_grade || "一般")}<br>
                予想時刻：${esc(r.fetched_at || "-")}<br>
                レース的中見込み：
                <b>
                  ${(Number(r.race_hit_probability || 0) * 100).toFixed(1)}%
                </b>
              </div>

              ${conditions}

              <div style="margin-top:8px">
                ${picksHtml}
              </div>

              ${
                r.result_combo
                  ? `<div style="margin-top:10px">
                      結果：
                      <b>${esc(r.result_combo)}</b>
                      / 払戻 ${yen(r.payout)}
                    </div>`
                  : ""
              }
            </div>
          </div>
        `;
      });

      if (!recentHtml) {
        recentHtml = `
          <div class="empty">
            このグレードの予想はまだありません
          </div>
        `;
      }

      const gradeContent =
        document.getElementById("gradeContent");

      gradeContent.innerHTML = `
        <div class="meta history">
          <b>${labels[grade]}・場ごとの成績</b>
          <div>${venueHtml}</div>
        </div>

        <div class="meta history">
          <b>${labels[grade]}・最近の予想</b>
          ${recentHtml}
        </div>
      `;

      resultBox
        .querySelectorAll(".grade-tab")
        .forEach((button) => {
          button.classList.toggle(
            "active",
            button.dataset.grade === grade
          );
        });

      gradeContent
        .querySelectorAll(".history-row")
        .forEach((row) => {
          if (!row.querySelector(".history-detail")) {
            return;
          }

          row.addEventListener("click", () => {
            row.classList.toggle("open");

            const arrow = row.querySelector(
              ".history-arrow"
            );

            if (arrow) {
              arrow.textContent =
                row.classList.contains("open")
                  ? "⌄"
                  : "›";
            }
          });
        });
    };

    resultBox.innerHTML = `
      <div class="stats-grid">
        <div class="stat">
          <small>保存予想</small>
          <strong>${all.saved_predictions || 0}</strong>
        </div>
        <div class="stat">
          <small>結果確認済み</small>
          <strong>${all.finished_races || 0}</strong>
        </div>
        <div class="stat">
          <small>的中レース</small>
          <strong>${all.hit_races || 0}</strong>
        </div>
        <div class="stat">
          <small>的中率</small>
          <strong>${pct(all.hit_rate)}</strong>
        </div>
        <div class="stat">
          <small>収支</small>
          <strong>${yen(all.total_profit)}</strong>
        </div>
        <div class="stat">
          <small>回収率</small>
          <strong>${pct(all.roi)}</strong>
        </div>
      </div>

      <div class="grade-tabs">
        ${Object.keys(labels).map((key) => `
          <button
            type="button"
            class="grade-tab"
            data-grade="${esc(key)}"
          >
            ${labels[key]}
          </button>
        `).join("")}
      </div>

      <div id="gradeContent"></div>
    `;

    resultBox
      .querySelectorAll(".grade-tab")
      .forEach((button) => {
        button.addEventListener("click", () => {
          renderGradeSection(
            button.dataset.grade || "SG"
          );
        });
      });

    renderGradeSection("SG");

    statusBox.textContent = "累計成績";

  } catch (err) {
    statusBox.textContent = "エラー";
    resultBox.innerHTML = `
      <div class="empty">${esc(err.message)}</div>
    `;
  }
});

refreshBtn.addEventListener("click", async () => {
  refreshBtn.disabled = true;
  statusBox.textContent = "結果を確認中…";
  resultBox.innerHTML = "";

  try {
    const res = await fetch("/check-results", {
      method: "POST"
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.message || data.error || "結果更新に失敗しました"
      );
    }

    const checked = Number(
      data.checked ?? data.updated ?? data.processed ?? 0
    );

    statusBox.textContent = "結果更新完了";

    resultBox.innerHTML = `
      <div class="empty">
        結果確認が完了しました
        ${checked ? `<br>${checked}件を確認・更新` : ""}
      </div>
    `;

    // 更新後に成績を自動で再表示
    await statsBtn.click();
  } catch (err) {
    statusBox.textContent = "エラー";
    resultBox.innerHTML = `
      <div class="empty">${esc(err.message)}</div>
    `;
  } finally {
    refreshBtn.disabled = false;
  }
});


deleteBtn.addEventListener("click", async () => {
  const ok = confirm(
    "予想ログ・結果・収支を全部削除します。元に戻せません。削除しますか？"
  );

  if (!ok) return;

  const secondCheck = confirm(
    "本当に全削除しますか？"
  );

  if (!secondCheck) return;

  statusBox.textContent = "削除中…";
  resultBox.innerHTML = "";

  try {
    const res = await fetch("/delete-all", {
      method: "POST"
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.message || data.error || "削除に失敗しました"
      );
    }

    statusBox.textContent = "全削除完了";
    resultBox.innerHTML = `
      <div class="empty">
        ${Number(data.deleted || 0)}件を削除しました
      </div>
    `;
  } catch (err) {
    statusBox.textContent = "エラー";
    resultBox.innerHTML = `
      <div class="empty">${esc(err.message)}</div>
    `;
  }
});

</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML_PAGE)

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
        result = predict_internal(race_text, date_text)
        result["log_id"] = save_prediction_log(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error":type(e).__name__, "message":str(e)}), 500



@app.route("/auto-predict", methods=["GET", "POST"])
def auto_predict():
    try:
        return jsonify(run_auto_predictions())
    except Exception as e:
        return jsonify({
            "error": type(e).__name__,
            "message": str(e),
        }), 500


@app.get("/stats")
def stats():
    try:
        return jsonify(build_stats())
    except Exception as e:
        return jsonify({
            "error": type(e).__name__,
            "message": str(e)
        }), 500

@app.route("/check-results", methods=["GET", "POST"])
def check_results():
    try:
        return jsonify(update_pending_results())
    except Exception as e:
        return jsonify({
            "error": type(e).__name__,
            "message": str(e)
        }), 500


@app.post("/delete-all")
def delete_all():
    try:
        deleted = delete_all_predictions()
        return jsonify({
            "status": "ok",
            "deleted": deleted
        })
    except Exception as e:
        return jsonify({
            "error": type(e).__name__,
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
