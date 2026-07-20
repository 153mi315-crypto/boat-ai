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
MODEL_DIR = BASE_DIR / 'models'
JST = ZoneInfo('Asia/Tokyo')
VENUES = {1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川', 6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国', 11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀', 16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松', 21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村'}
VENUE_ALIASES = {v: k for k, v in VENUES.items()}
VENUE_ALIASES.update({'琵琶湖': 11, 'びわ湖': 11, 'からつ': 23, 'おおむら': 24})
HEADERS = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15', 'Accept-Language': 'ja-JP,ja;q=0.9'}
FIRST_MODEL_PATH = MODEL_DIR / 'first_model.joblib'
SECOND_MODEL_PATH = MODEL_DIR / 'second_conditional_model.joblib'
THIRD_MODEL_PATH = MODEL_DIR / 'third_conditional_model.joblib'
SCHEMA_PATH = MODEL_DIR / 'feature_schema.json'
SHADOW_MODEL_DIR = BASE_DIR / 'models_new'
SHADOW_FIRST_MODEL_PATH = SHADOW_MODEL_DIR / 'first_model.joblib'
SHADOW_SECOND_MODEL_PATH = SHADOW_MODEL_DIR / 'second_conditional_model.joblib'
SHADOW_THIRD_MODEL_PATH = SHADOW_MODEL_DIR / 'third_conditional_model.joblib'
SHADOW_SCHEMA_PATH = SHADOW_MODEL_DIR / 'feature_schema.json'
SHADOW_MODEL_VERSION = 'v1.6'
_loaded = None
_shadow_loaded = None

def load_assets():
    global _loaded
    if _loaded is not None:
        return _loaded
    required = [FIRST_MODEL_PATH, SECOND_MODEL_PATH, THIRD_MODEL_PATH, SCHEMA_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing model files: ' + ', '.join(missing))
    first_model = joblib.load(FIRST_MODEL_PATH)
    second_model = joblib.load(SECOND_MODEL_PATH)
    third_model = joblib.load(THIRD_MODEL_PATH)
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    _loaded = {'first_model': first_model, 'second_model': second_model, 'third_model': third_model, 'schema': schema}
    return _loaded

def load_shadow_assets():
    global _shadow_loaded
    if _shadow_loaded is not None:
        return _shadow_loaded
    required = [SHADOW_FIRST_MODEL_PATH, SHADOW_SECOND_MODEL_PATH, SHADOW_THIRD_MODEL_PATH, SHADOW_SCHEMA_PATH]
    if not all((path.exists() for path in required)):
        return None
    try:
        with open(SHADOW_SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        _shadow_loaded = {'first_model': joblib.load(SHADOW_FIRST_MODEL_PATH), 'second_model': joblib.load(SHADOW_SECOND_MODEL_PATH), 'third_model': joblib.load(SHADOW_THIRD_MODEL_PATH), 'schema': schema}
        return _shadow_loaded
    except Exception:
        return None

def predict_combo_table(race_source, assets):
    first_model = assets['first_model']
    second_model = assets['second_model']
    third_model = assets['third_model']
    schema = assets['schema']
    feature_cols = schema['feature_cols']
    categorical_cols = schema['categorical_cols']
    context_cols = schema['context_cols']
    second_cols = schema['second_cols']
    third_cols = schema['third_cols']
    x_cols = feature_cols + categorical_cols
    race = build_features(race_source.copy(), schema).sort_values('lane').copy()
    raw1 = first_model.predict_proba(race[x_cols])[:, 1]
    race['p1'] = raw1 / raw1.sum()
    rows = []
    for first_lane in range(1, 7):
        winner = race[race['lane'] == first_lane].iloc[0]
        sec = race[race['lane'] != first_lane].copy()
        for c in context_cols:
            sec[f'winner_{c}'] = winner[c]
            sec[f'candidate_minus_winner_{c}'] = sec[c] - winner[c]
        raw2 = second_model.predict_proba(sec[second_cols])[:, 1]
        sec['p2'] = raw2 / raw2.sum()
        for _, second in sec.iterrows():
            second_lane = int(second['lane'])
            third = race[~race['lane'].isin([first_lane, second_lane])].copy()
            for c in context_cols:
                third[f'winner_{c}'] = winner[c]
                third[f'second_{c}'] = second[c]
                third[f'candidate_minus_winner_{c}'] = third[c] - winner[c]
                third[f'candidate_minus_second_{c}'] = third[c] - second[c]
            raw3 = third_model.predict_proba(third[third_cols])[:, 1]
            third['p3'] = raw3 / raw3.sum()
            for _, r3 in third.iterrows():
                rows.append({'combo': f"{first_lane}-{second_lane}-{int(r3['lane'])}", 'probability': float(winner['p1']) * float(second['p2']) * float(r3['p3'])})
    pred = pd.DataFrame(rows)
    pred['probability'] /= pred['probability'].sum()
    return pred

def parse_race_input(text: str):
    s = re.sub('\\s+', '', str(text))
    m = re.fullmatch('(.+?)(\\d{1,2})R?', s, flags=re.I)
    if not m:
        raise ValueError('race must look like 蒲郡12R')
    venue_name = m.group(1)
    race_no = int(m.group(2))
    if venue_name not in VENUE_ALIASES:
        raise ValueError(f'unknown venue: {venue_name}')
    if not 1 <= race_no <= 12:
        raise ValueError('race number must be 1-12')
    return (VENUE_ALIASES[venue_name], race_no)

def flatten_columns(df):
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ['_'.join((str(x) for x in col if str(x) != 'nan')).strip('_') for col in out.columns]
    else:
        out.columns = [str(c) for c in out.columns]
    return out

def num(x):
    if pd.isna(x):
        return np.nan
    m = re.search('-?\\d+(?:\\.\\d+)?', str(x).replace(',', ''))
    return float(m.group()) if m else np.nan

def get_html(page, hd, venue_code, race_no):
    url = f'https://www.boatrace.jp/owpc/pc/race/{page}'
    params = {'hd': hd, 'jcd': f'{venue_code:02d}', 'rno': race_no}
    res = requests.get(url, params=params, headers=HEADERS, timeout=30)
    res.raise_for_status()
    res.encoding = res.apparent_encoding or 'utf-8'
    return (res.text, res.url)

def find_main_racelist_table(html):
    tables = [flatten_columns(t) for t in pd.read_html(StringIO(html))]
    candidates = []
    for t in tables:
        text = ' '.join(map(str, t.columns)) + ' ' + ' '.join(t.astype(str).head(10).fillna('').values.ravel())
        score = sum((k in text for k in ['登録番号', '全国', '当地', 'モーター', 'ボート']))
        if len(t) >= 6:
            candidates.append((score, len(t.columns), t))
    if not candidates:
        raise RuntimeError('racelist table not found')
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]

def parse_racelist(html, target_date, venue_code, race_no):
    t = find_main_racelist_table(html).iloc[:6].reset_index(drop=True)
    rows = []
    for i, row in t.iterrows():
        vals = [str(v).strip() for v in row.tolist()]
        joined = ' | '.join(vals)
        reg_class = re.search('(\\d{4})\\s*/?\\s*(A1|A2|B1|B2)', joined)
        if reg_class:
            registration = int(reg_class.group(1))
            klass = reg_class.group(2)
        else:
            rev = re.search('(A1|A2|B1|B2)\\s*/?\\s*(\\d{4})', joined)
            if not rev:
                raise RuntimeError(f'registration parse failed lane={i + 1}')
            klass = rev.group(1)
            registration = int(rev.group(2))
        name = f'{i + 1}号艇'
        for v in vals:
            if re.search('[\\u3040-\\u30ff\\u3400-\\u9fff]', v) and (not any((k in v for k in ['全国', '当地', 'モーター', 'ボート', '成績', '支部', '出身']))):
                cleaned = re.sub('\\s+', '', v)
                if 2 <= len(cleaned) <= 12:
                    name = cleaned
                    break
        branch = '不明'
        for v in vals:
            m = re.search('([\\u3400-\\u9fff]{1,4})\\s*/\\s*([\\u3400-\\u9fff]{1,4})', v)
            if m:
                branch = m.group(1)
                break
        age_m = re.search('(\\d{2})歳', joined)
        weight_m = re.search('(\\d{2}(?:\\.\\d)?)kg', joined)

        def by_keywords(include):
            for c, v in row.items():
                cs = str(c)
                if all((k in cs for k in include)):
                    value = num(v)
                    if not pd.isna(value):
                        return value
            return np.nan
        rows.append({'date': target_date.isoformat(), 'venue_code': venue_code, 'venue': VENUES[venue_code], 'race_no': race_no, 'race_title_program': '当日公式', 'lane': i + 1, 'registration': registration, 'player_name_program': name, 'age': int(age_m.group(1)) if age_m else np.nan, 'branch': branch, 'weight': float(weight_m.group(1)) if weight_m else np.nan, 'class': klass, 'national_win_rate': by_keywords(['全国', '勝率']), 'national_2rate': by_keywords(['全国', '2連率']), 'local_win_rate': by_keywords(['当地', '勝率']), 'local_2rate': by_keywords(['当地', '2連率']), 'motor_no_program': by_keywords(['モーター', 'No']), 'motor_2rate': by_keywords(['モーター', '2連率']), 'boat_no_program': by_keywords(['ボート', 'No']), 'boat_2rate': by_keywords(['ボート', '2連率'])})
    race = pd.DataFrame(rows)
    if len(race) != 6:
        raise RuntimeError(f'expected 6 boats, got {len(race)}')
    return race

def find_col(df, keywords, exclude=()):
    for c in df.columns:
        name = re.sub('\\s+', '', str(c))
        if all((k in name for k in keywords)) and (not any((x in name for x in exclude))):
            return c
    return None

def numeric_series(series):
    return pd.to_numeric(series.astype(str).str.extract('(-?\\d+(?:\\.\\d+)?)', expand=False), errors='coerce')

def parse_beforeinfo(html):
    tables = [flatten_columns(t) for t in pd.read_html(StringIO(html))]
    scored = []
    for t in tables:
        text = ' '.join(map(str, t.columns)) + ' ' + ' '.join(t.astype(str).head(10).fillna('').values.ravel())
        score = sum((k in text for k in ['展示', 'チルト', '体重', '部品交換']))
        if len(t) >= 6:
            scored.append((score, len(t.columns), t))
    if not scored:
        raise RuntimeError('beforeinfo table not found')
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    b = scored[0][2].head(6).reset_index(drop=True)
    b.columns = [re.sub('\\s+', '', str(c)) for c in b.columns]
    time_col = find_col(b, ['展示', 'タイム']) or find_col(b, ['展示'], exclude=('ST',))
    tilt_col = find_col(b, ['チルト'])
    weight_col = find_col(b, ['体重'])
    parts_col = find_col(b, ['部品']) or find_col(b, ['交換'])
    out = pd.DataFrame({'lane': np.arange(1, 7)})
    out['exhibition_time'] = numeric_series(b[time_col]) if time_col else np.nan
    out['tilt'] = numeric_series(b[tilt_col]) if tilt_col else np.nan
    out['weight_before'] = numeric_series(b[weight_col]) if weight_col else np.nan
    out['parts_exchange'] = b[parts_col].astype(str).replace('nan', '') if parts_col else ''
    out['exhibition_st'] = np.nan
    for t in tables:
        t2 = t.copy()
        t2.columns = [re.sub('\\s+', '', str(c)) for c in t2.columns]
        st_col = find_col(t2, ['ST'])
        if st_col is None:
            continue
        values = numeric_series(t2[st_col])
        if values.between(-0.3, 1.5).sum() >= 4:
            out['exhibition_st'] = values.head(6).reset_index(drop=True)
            break
    if out['exhibition_time'].notna().sum() < 4:
        for c in b.columns:
            values = numeric_series(b[c])
            if values.between(6.0, 8.0).sum() >= 4:
                out['exhibition_time'] = values
                break
    out['parts_exchange_flag'] = out['parts_exchange'].fillna('').astype(str).str.strip().ne('').astype(int)
    return out

def build_features(race, schema):
    feature_cols = schema['feature_cols']
    categorical_cols = schema['categorical_cols']
    race = race.copy()
    race['date'] = pd.to_datetime(race['date'])
    race['month'] = race['date'].dt.month
    race['dayofweek'] = race['date'].dt.dayofweek
    race['is_weekend'] = race['dayofweek'].isin([5, 6]).astype(int)
    numeric_defaults = {'registration': 0, 'age': 0, 'weight': 0, 'national_win_rate': 0, 'national_2rate': 0, 'local_win_rate': 0, 'local_2rate': 0, 'motor_no_program': 0, 'motor_2rate': 0, 'boat_no_program': 0, 'boat_2rate': 0, 'exhibition_time': 0, 'exhibition_st': 0, 'tilt': 0, 'weight_before': 0, 'parts_exchange_flag': 0}
    for c, default in numeric_defaults.items():
        if c not in race.columns:
            race[c] = default
        race[c] = pd.to_numeric(race[c], errors='coerce')
        med = race[c].median()
        if pd.isna(med):
            med = default
        race[c] = race[c].fillna(med)
    relative_cols = ['national_win_rate', 'national_2rate', 'local_win_rate', 'local_2rate', 'motor_2rate', 'boat_2rate', 'exhibition_time', 'exhibition_st', 'tilt', 'weight_before']
    for c in relative_cols:
        race[f'{c}_race_diff'] = race[c] - race[c].mean()
        ascending = c in ['exhibition_time', 'exhibition_st', 'weight_before']
        race[f'{c}_race_rank'] = race[c].rank(method='average', ascending=ascending)
    race['lane_x_national_win'] = race['lane'] * race['national_win_rate']
    race['lane_x_local_win'] = race['lane'] * race['local_win_rate']
    race['lane_x_motor'] = race['lane'] * race['motor_2rate']
    race['lane_x_exhibition'] = race['lane'] * race['exhibition_time']
    for c in categorical_cols:
        if c not in race.columns:
            race[c] = '不明'
        race[c] = race[c].fillna('不明').astype('category')
    missing = [c for c in feature_cols if c not in race.columns]
    if missing:
        raise KeyError('missing features: ' + ', '.join(missing))
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
                group = part.iloc[:, (first - 1) * 3:first * 3].copy()
                group.columns = ['second', 'third', 'odds']
                group['second'] = pd.to_numeric(group['second'], errors='coerce').ffill()
                group['third'] = pd.to_numeric(group['third'], errors='coerce')
                group['odds'] = pd.to_numeric(group['odds'].astype(str).str.replace(',', '', regex=False), errors='coerce')
                for row in group.itertuples(index=False):
                    if pd.isna(row.second) or pd.isna(row.third) or pd.isna(row.odds):
                        continue
                    second, third, odd = (int(row.second), int(row.third), float(row.odds))
                    if 1 <= second <= 6 and 1 <= third <= 6 and (len({first, second, third}) == 3) and (odd > 0):
                        parsed.append({'combo': f'{first}-{second}-{third}', 'odds': odd})
            result = pd.DataFrame(parsed).drop_duplicates('combo')
            if len(result) > len(best):
                best = result
    if best.empty:
        raise RuntimeError('odds table parse failed')
    return best

def allocate_stakes_by_odds(recommendations, budget=1000, unit=100, target_payout=20000, min_stake=100):
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
    if len(recommendations) <= 4:
        allocated = []
        for rec in recommendations:
            odds = max(float(rec['odds']), 1.01)
            stake_units = math.ceil(target_payout / odds / unit)
            stake = max(min_stake, int(stake_units * unit))
            allocated.append({**rec, 'stake': stake})
        allocation_mode = 'minimum_target_payout'
    else:
        max_picks = budget // unit
        recommendations = recommendations[:max_picks]
        count = len(recommendations)
        odds = np.array([max(float(rec['odds']), 1.01) for rec in recommendations], dtype=float)
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
            order = np.argsort(-(raw_extra - floor_extra))
            for idx in order[:leftover]:
                units[idx] += 1
        allocated = [{**rec, 'stake': int(stake_units * unit)} for rec, stake_units in zip(recommendations, units)]
        allocation_mode = 'full_budget'
    total_stake = sum((int(item['stake']) for item in allocated))
    for item in allocated:
        odds = float(item['odds'])
        stake = int(item['stake'])
        gross_if_hit = odds * stake
        item['gross_if_hit'] = round(gross_if_hit)
        item['net_if_hit'] = round(gross_if_hit - total_stake)
        item['race_budget'] = budget
        item['total_stake'] = total_stake
        item['target_payout'] = target_payout
        item['allocation_mode'] = allocation_mode
    allocated.sort(key=lambda item: (float(item.get('safe_EV') or 0), float(item.get('safe_probability') or 0)), reverse=True)
    return allocated

def fetch_active_venue_codes(target_date):
    """公式の本日のレース一覧から開催中の場コードを取得する。"""
    hd = target_date.strftime('%Y%m%d')
    url = 'https://www.boatrace.jp/owpc/pc/race/index'
    res = requests.get(url, params={'hd': hd}, headers=HEADERS, timeout=30)
    res.raise_for_status()
    res.encoding = res.apparent_encoding or 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')
    codes = set()
    for link in soup.find_all('a', href=True):
        href = str(link.get('href', ''))
        if 'racelist' not in href:
            continue
        match = re.search('(?:\\?|&)jcd=(\\d{1,2})', href)
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
            values = [str(value).strip() for value in row.tolist() if not pd.isna(value)]
            joined = ' '.join(values)
            if '締切予定時刻' not in joined:
                continue
            times = re.findall('(?<!\\d)([01]?\\d|2[0-3]):([0-5]\\d)(?!\\d)', joined)
            formatted = [f'{int(h):02d}:{m}' for h, m in times]
            if len(formatted) >= 1:
                return formatted[:12]
    page_text = BeautifulSoup(html, 'html.parser').get_text(' ', strip=True)
    match = re.search('締切予定時刻(.{0,500})', page_text)
    if match:
        times = re.findall('(?<!\\d)([01]?\\d|2[0-3]):([0-5]\\d)(?!\\d)', match.group(1))
        formatted = [f'{int(h):02d}:{m}' for h, m in times]
        if formatted:
            return formatted[:12]
    raise RuntimeError('締切予定時刻を取得できませんでした')

def fetch_venue_deadlines(target_date, venue_code):
    hd = target_date.strftime('%Y%m%d')
    html, _ = get_html('racelist', hd, venue_code, 1)
    times = parse_deadline_times(html)
    deadlines = []
    for index, time_text in enumerate(times, start=1):
        hour, minute = map(int, time_text.split(':'))
        deadline = datetime(target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=JST)
        deadlines.append({'venue_code': venue_code, 'venue': VENUES[venue_code], 'race_no': index, 'deadline': deadline})
    return deadlines

def already_auto_predicted(target_date, venue_code, race_no):
    doc_id = f'{target_date.isoformat()}_{int(venue_code):02d}_{int(race_no):02d}'
    snap = get_firestore().collection('predictions').document(doc_id).get()
    if not snap.exists:
        return False
    data = snap.to_dict() or {}
    return data.get('source') == 'auto'

def run_auto_predictions(now=None, min_minutes_before=3, max_minutes_before=7, max_races=8):
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
            deadlines = fetch_venue_deadlines(target_date, venue_code)
            for race in deadlines:
                minutes_before = (race['deadline'] - now).total_seconds() / 60
                if min_minutes_before <= minutes_before <= max_minutes_before:
                    due_races.append({**race, 'minutes_before': round(minutes_before, 1)})
        except Exception as exc:
            schedule_errors.append({'venue': VENUES.get(venue_code, str(venue_code)), 'error': f'{type(exc).__name__}: {exc}'})
    due_races.sort(key=lambda item: item['deadline'])
    due_races = due_races[:max_races]
    predicted = []
    skipped = []
    prediction_errors = []
    for race in due_races:
        venue_code = race['venue_code']
        race_no = race['race_no']
        race_text = f'{VENUES[venue_code]}{race_no}R'
        if already_auto_predicted(target_date, venue_code, race_no):
            skipped.append({'race': race_text, 'reason': 'already_predicted'})
            continue
        try:
            result = predict_internal(race_text, target_date.isoformat())
            log_id = save_prediction_log(result, source='auto')
            predicted.append({'race': race_text, 'deadline': race['deadline'].isoformat(), 'minutes_before': race['minutes_before'], 'decision': '推奨' if result.get('recommendations') else '見送り', 'recommendations': len(result.get('recommendations', [])), 'log_id': log_id})
        except Exception as exc:
            prediction_errors.append({'race': race_text, 'error': f'{type(exc).__name__}: {exc}'})
    return {'status': 'ok', 'checked_at': now.isoformat(), 'active_venues': len(venue_codes), 'due_races': len(due_races), 'predicted': predicted, 'skipped': skipped, 'schedule_errors': schedule_errors[:10], 'prediction_errors': prediction_errors[:10]}

def predict_shadow_only(race_text: str, date_text: str | None=None):
    shadow_assets = load_shadow_assets()
    if shadow_assets is None:
        raise RuntimeError('models_new unavailable')
    venue_code, race_no = parse_race_input(race_text)
    target_date = date.fromisoformat(date_text) if date_text else datetime.now(JST).date()
    hd = target_date.strftime('%Y%m%d')
    racelist_html, _ = get_html('racelist', hd, venue_code, race_no)
    before_html, _ = get_html('beforeinfo', hd, venue_code, race_no)
    odds_html, _ = get_html('odds3t', hd, venue_code, race_no)
    race = parse_racelist(racelist_html, target_date, venue_code, race_no)
    before = parse_beforeinfo(before_html)
    race = race.merge(before, on='lane', how='left', validate='one_to_one')
    for column in ['exhibition_time', 'exhibition_st', 'tilt', 'weight_before']:
        race[column] = pd.to_numeric(race[column], errors='coerce')
        median = race[column].median()
        race[column] = race[column].fillna(0 if pd.isna(median) else median)
    prediction = predict_combo_table(race, shadow_assets)
    odds = parse_odds_table(odds_html)
    shadow_out = prediction.merge(odds, on='combo', how='left')
    shadow_out['EV'] = shadow_out['probability'] * shadow_out['odds']
    shadow_out = shadow_out.sort_values(['probability', 'EV'], ascending=False).reset_index(drop=True)
    fetched_at = datetime.now(JST)
    return {'status': 'ok', 'model_version': SHADOW_MODEL_VERSION, 'error': None, 'fetched_at': fetched_at.isoformat(), 'all_combo_predictions': [{'combo': row.combo, 'probability': round(float(row.probability), 8), 'odds': None if pd.isna(row.odds) else round(float(row.odds), 1), 'EV': None if pd.isna(row.EV) else round(float(row.EV), 4), 'rank': int(rank)} for rank, row in enumerate(shadow_out.itertuples(), start=1)]}

def refresh_due_shadow_predictions(now=None, min_minutes_before=0.05, max_minutes_before=2.5, max_races=12):
    now = now or datetime.now(JST)
    target_date = now.date()
    venue_codes = fetch_active_venue_codes(target_date)
    due_races = []
    schedule_errors = []
    for venue_code in venue_codes:
        try:
            deadlines = fetch_venue_deadlines(target_date, venue_code)
            for race in deadlines:
                minutes_before = (race['deadline'] - now).total_seconds() / 60
                if min_minutes_before <= minutes_before <= max_minutes_before:
                    due_races.append({**race, 'minutes_before': round(minutes_before, 3)})
        except Exception as exc:
            schedule_errors.append({'venue': VENUES.get(venue_code, str(venue_code)), 'error': f'{type(exc).__name__}: {exc}'})
    due_races.sort(key=lambda item: item['deadline'])
    due_races = due_races[:max_races]
    updated = []
    skipped = []
    errors = []
    db = get_firestore()

    def parse_saved_datetime(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=JST)
        return parsed.astimezone(JST)
    for race in due_races:
        venue_code = int(race['venue_code'])
        race_no = int(race['race_no'])
        race_text = f'{VENUES[venue_code]}{race_no}R'
        doc_id = f'{target_date.isoformat()}_{venue_code:02d}_{race_no:02d}'
        doc_ref = db.collection('predictions').document(doc_id)
        snap = doc_ref.get()
        if not snap.exists:
            skipped.append({'race': race_text, 'reason': 'base_prediction_missing'})
            continue
        existing = snap.to_dict() or {}
        existing_shadow = existing.get('shadow_model') or {}
        previous_value = existing_shadow.get('fetched_at') or existing.get('shadow_fetched_at')
        try:
            previous_dt = parse_saved_datetime(previous_value)
        except Exception:
            previous_dt = None
        if previous_dt is not None:
            elapsed_seconds = (now - previous_dt).total_seconds()
            if elapsed_seconds < 45:
                skipped.append({'race': race_text, 'reason': 'refreshed_within_45_seconds', 'elapsed_seconds': round(elapsed_seconds, 1)})
                continue
        try:
            shadow_result = predict_shadow_only(race_text, target_date.isoformat())
            shadow_result['near_deadline_refresh'] = True
            shadow_result['minutes_before_deadline'] = race['minutes_before']
            snapshots = existing.get('shadow_odds_snapshots')
            if not isinstance(snapshots, list):
                snapshots = []
            snapshot = {'fetched_at': shadow_result.get('fetched_at'), 'minutes_before_deadline': race['minutes_before'], 'all_combo_predictions': shadow_result.get('all_combo_predictions', [])}
            snapshots = (snapshots + [snapshot])[-5:]
            saved_at = datetime.now(JST)
            doc_ref.set({'shadow_model': shadow_result, 'shadow_fetched_at': saved_at, 'shadow_minutes_before_deadline': race['minutes_before'], 'shadow_odds_snapshots': snapshots, 'shadow_refresh_count': len(snapshots)}, merge=True)
            updated.append({'race': race_text, 'deadline': race['deadline'].isoformat(), 'minutes_before': race['minutes_before'], 'snapshot_count': len(snapshots), 'status': 'ok'})
        except Exception as exc:
            errors.append({'race': race_text, 'error': f'{type(exc).__name__}: {exc}'})
    return {'status': 'ok', 'checked_at': now.isoformat(), 'window_minutes': [min_minutes_before, max_minutes_before], 'due_races': len(due_races), 'updated': updated, 'skipped': skipped, 'schedule_errors': schedule_errors[:10], 'prediction_errors': errors[:10]}

def normalize_grade_token(value):
    value = str(value or '').upper()
    value = value.replace('Ｓ', 'S').replace('Ｇ', 'G').replace('Ⅰ', '1').replace('Ⅱ', '2').replace('Ⅲ', '3').replace('１', '1').replace('２', '2').replace('３', '3')
    if re.search('(^|[^A-Z0-9])SG([^A-Z0-9]|$)', value):
        return 'SG'
    if re.search('(^|[^A-Z0-9])G1([^A-Z0-9]|$)', value):
        return 'G1'
    if re.search('(^|[^A-Z0-9])G2([^A-Z0-9]|$)', value):
        return 'G2'
    if re.search('(^|[^A-Z0-9])G3([^A-Z0-9]|$)', value):
        return 'G3'
    if '一般' in value:
        return '一般'
    return None

def target_date_in_range(target_date, start_month, start_day, end_month, end_day):
    """
    月間スケジュールの MM/DD - MM/DD が対象日を含むか判定。
    年またぎにも対応する。
    """
    year = target_date.year
    start = date(year, int(start_month), int(start_day))
    end = date(year, int(end_month), int(end_day))
    if end < start:
        end = date(year + 1, int(end_month), int(end_day))
        if target_date < start:
            target_date = date(year + 1, target_date.month, target_date.day)
    return start <= target_date <= end

def grade_from_schedule_block(block):
    """
    開催ブロック内の表示・画像alt・src・classから
    公式表示のグレードを取得。
    """
    signals = [block.get_text(' ', strip=True), ' '.join(block.get('class', [])), str(block.get('id', ''))]
    for node in block.find_all(True):
        signals.extend([str(node.get('alt', '')), str(node.get('src', '')), str(node.get('class', '')), str(node.get('id', '')), str(node.get('title', ''))])
    joined = ' '.join(signals)
    for pattern, grade in [('(?:GRADE|ICON|LABEL)[_\\-/ ]*SG', 'SG'), ('(?:GRADE|ICON|LABEL)[_\\-/ ]*G1', 'G1'), ('(?:GRADE|ICON|LABEL)[_\\-/ ]*G2', 'G2'), ('(?:GRADE|ICON|LABEL)[_\\-/ ]*G3', 'G3')]:
        if re.search(pattern, joined, re.I):
            return grade
    return normalize_grade_token(joined)

def fetch_official_schedule_grade(target_date, venue_code):
    """
    BOAT RACE公式の月間スケジュールを参照し、
    日付＋会場が一致する開催の公式グレードを返す。
    """
    venue_code = int(venue_code)
    venue_name = VENUES[venue_code]
    ym = target_date.strftime('%Y%m')
    url = 'https://www.boatrace.jp/owpc/pc/race/monthlyschedule'
    response = requests.get(url, params={'ym': ym}, headers=HEADERS, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    def compact(value):
        return re.sub('\\s+', '', str(value or ''))
    venue_compact = compact(venue_name)
    candidates = []
    for node in soup.find_all(string=lambda s: s is not None and venue_compact in compact(s)):
        current = node.parent
        for _ in range(8):
            if current is None:
                break
            block_text = current.get_text(' ', strip=True)
            if len(block_text) >= len(venue_name) and re.search('\\d{1,2}/\\d{1,2}\\s*[-–—〜～]\\s*\\d{1,2}/\\d{1,2}', block_text):
                candidates.append(current)
            current = current.parent
    candidates = sorted({id(block): block for block in candidates}.values(), key=lambda block: len(block.get_text(' ', strip=True)))
    range_pattern = re.compile('(\\d{1,2})/(\\d{1,2})\\s*[-–—〜～]\\s*(\\d{1,2})/(\\d{1,2})')
    for venue_block in candidates:
        event_blocks = venue_block.find_all(['li', 'tr', 'article', 'section', 'div'], recursive=True)
        event_blocks.insert(0, venue_block)
        for event_block in event_blocks:
            block_text = event_block.get_text(' ', strip=True)
            match = range_pattern.search(block_text)
            if not match:
                continue
            if not target_date_in_range(target_date, *match.groups()):
                continue
            grade = grade_from_schedule_block(event_block)
            if grade:
                return grade
    raw_html = response.text
    venue_positions = [match.start() for match in re.finditer(re.escape(venue_name), raw_html)]
    for position in venue_positions:
        fragment = raw_html[position:position + 30000]
        fragment_soup = BeautifulSoup(fragment, 'html.parser')
        for block in fragment_soup.find_all(['li', 'tr', 'article', 'section', 'div']):
            block_text = block.get_text(' ', strip=True)
            match = range_pattern.search(block_text)
            if not match:
                continue
            if target_date_in_range(target_date, *match.groups()):
                grade = grade_from_schedule_block(block)
                if grade:
                    return grade
    return None

def resolve_race_grade(target_date, venue_code, fallback_html=''):
    """
    1. 月間スケジュールの公式表示
    2. レースページのタイトル判定
    の順でグレードを決定。
    """
    try:
        official_grade = fetch_official_schedule_grade(target_date, venue_code)
        if official_grade:
            return official_grade
    except Exception:
        pass
    return detect_race_grade(fallback_html)

def detect_race_grade(html):
    """
    大会タイトル周辺だけからSG/G1/G2/G3/一般を判定する。

    ページ全体には他グレード大会へのリンクや画像が含まれるため、
    全HTMLを検索対象にしない。
    """
    soup = BeautifulSoup(html, 'html.parser')
    title_parts = []
    if soup.title:
        title_parts.append(soup.title.get_text(' ', strip=True))
    selectors = ['h1', 'h2', 'h3', 'h4', '[class*="title"]', '[class*="heading"]', '[class*="event"]', '[class*="raceName"]', '[id*="title"]', '[id*="heading"]', '[id*="event"]']
    seen = set()
    for selector in selectors:
        for node in soup.select(selector):
            value = node.get_text(' ', strip=True)
            if value and value not in seen:
                seen.add(value)
                title_parts.append(value)
            for img in node.find_all('img'):
                alt = str(img.get('alt', '')).strip()
                if alt and alt not in seen:
                    seen.add(alt)
                    title_parts.append(alt)
    event_text = ' '.join(title_parts).upper()
    compact = re.sub('\\s+', '', event_text)
    sg_keywords = ['ボートレースクラシック', 'ボートレースオールスター', 'グランドチャンピオン', 'オーシャンカップ', 'ボートレースメモリアル', 'ボートレースダービー', 'チャレンジカップ', 'グランプリ']
    g2_keywords = ['モーターボート大賞', 'レディースオールスター', '全国ボートレース甲子園', 'モーターボート誕生祭', '秩父宮妃記念杯']
    if any((keyword.upper() in event_text for keyword in g2_keywords)):
        return 'G2'
    if any((keyword.upper() in event_text for keyword in sg_keywords)):
        return 'SG'
    if 'GⅢ' in event_text or 'ＧⅢ' in event_text or re.search('(^|[^A-Z0-9])G3([^A-Z0-9]|$)', event_text) or ('GRADE_G3' in compact) or ('ICON_G3' in compact):
        return 'G3'
    if 'GⅡ' in event_text or 'ＧⅡ' in event_text or re.search('(^|[^A-Z0-9])G2([^A-Z0-9]|$)', event_text) or ('GRADE_G2' in compact) or ('ICON_G2' in compact):
        return 'G2'
    if 'GⅠ' in event_text or 'ＧⅠ' in event_text or re.search('(^|[^A-Z0-9])G1([^A-Z0-9]|$)', event_text) or ('GRADE_G1' in compact) or ('ICON_G1' in compact):
        return 'G1'
    if re.search('(^|[^A-Z0-9])SG([^A-Z0-9]|$)', event_text) or 'GRADE_SG' in compact or 'ICON_SG' in compact:
        return 'SG'
    return '一般'

def predict_internal(race_text: str, date_text: str | None=None):
    assets = load_assets()
    first_model = assets['first_model']
    second_model = assets['second_model']
    third_model = assets['third_model']
    schema = assets['schema']
    feature_cols = schema['feature_cols']
    categorical_cols = schema['categorical_cols']
    context_cols = schema['context_cols']
    second_cols = schema['second_cols']
    third_cols = schema['third_cols']
    x_cols = feature_cols + categorical_cols
    venue_code, race_no = parse_race_input(race_text)
    target_date = date.fromisoformat(date_text) if date_text else datetime.now(JST).date()
    hd = target_date.strftime('%Y%m%d')
    racelist_html, _ = get_html('racelist', hd, venue_code, race_no)
    before_html, _ = get_html('beforeinfo', hd, venue_code, race_no)
    odds_html, _ = get_html('odds3t', hd, venue_code, race_no)
    race = parse_racelist(racelist_html, target_date, venue_code, race_no)
    before = parse_beforeinfo(before_html)
    race = race.merge(before, on='lane', how='left', validate='one_to_one')
    for c in ['exhibition_time', 'exhibition_st', 'tilt', 'weight_before']:
        race[c] = pd.to_numeric(race[c], errors='coerce')
        med = race[c].median()
        race[c] = race[c].fillna(0 if pd.isna(med) else med)
    race_source = race.copy()
    race = build_features(race, schema).sort_values('lane').copy()
    raw1 = first_model.predict_proba(race[x_cols])[:, 1]
    race['p1'] = raw1 / raw1.sum()
    rows = []
    for first_lane in range(1, 7):
        winner = race[race['lane'] == first_lane].iloc[0]
        sec = race[race['lane'] != first_lane].copy()
        for c in context_cols:
            sec[f'winner_{c}'] = winner[c]
            sec[f'candidate_minus_winner_{c}'] = sec[c] - winner[c]
        raw2 = second_model.predict_proba(sec[second_cols])[:, 1]
        sec['p2'] = raw2 / raw2.sum()
        for _, second in sec.iterrows():
            second_lane = int(second['lane'])
            third = race[~race['lane'].isin([first_lane, second_lane])].copy()
            for c in context_cols:
                third[f'winner_{c}'] = winner[c]
                third[f'second_{c}'] = second[c]
                third[f'candidate_minus_winner_{c}'] = third[c] - winner[c]
                third[f'candidate_minus_second_{c}'] = third[c] - second[c]
            raw3 = third_model.predict_proba(third[third_cols])[:, 1]
            third['p3'] = raw3 / raw3.sum()
            for _, r3 in third.iterrows():
                rows.append({'combo': f"{first_lane}-{second_lane}-{int(r3['lane'])}", 'probability': float(winner['p1']) * float(second['p2']) * float(r3['p3'])})
    pred = pd.DataFrame(rows)
    pred['probability'] /= pred['probability'].sum()
    odds = parse_odds_table(odds_html)
    out = pred.merge(odds, on='combo', how='left')
    temperature = 1.8
    p = out['probability'].clip(lower=1e-12).to_numpy()
    scaled = np.exp(np.log(p) / temperature)
    scaled /= scaled.sum()
    out['calibrated_probability'] = scaled
    out['market_raw_probability'] = 1 / out['odds']
    out['market_probability'] = out['market_raw_probability'] / out['market_raw_probability'].sum()
    out['safe_probability'] = 0.75 * out['calibrated_probability'] + 0.25 * out['market_probability']
    out['safe_probability'] /= out['safe_probability'].sum()
    out['safe_EV'] = out['safe_probability'] * out['odds']
    out['market_disagreement'] = out['safe_probability'] / out['market_probability']
    shadow_result = {'status': 'unavailable', 'model_version': SHADOW_MODEL_VERSION, 'error': None, 'all_combo_predictions': []}
    try:
        shadow_assets = load_shadow_assets()
        if shadow_assets is None:
            shadow_result['error'] = 'models_new unavailable'
        else:
            shadow_pred = predict_combo_table(race_source, shadow_assets)
            shadow_out = shadow_pred.merge(odds, on='combo', how='left')
            shadow_out['EV'] = shadow_out['probability'] * shadow_out['odds']
            shadow_out = shadow_out.sort_values(['probability', 'EV'], ascending=False).reset_index(drop=True)
            shadow_result = {'status': 'ok', 'model_version': SHADOW_MODEL_VERSION, 'error': None, 'all_combo_predictions': [{'combo': row.combo, 'probability': round(float(row.probability), 8), 'odds': None if pd.isna(row.odds) else round(float(row.odds), 1), 'EV': None if pd.isna(row.EV) else round(float(row.EV), 4), 'rank': int(rank)} for rank, row in enumerate(shadow_out.itertuples(), start=1)]}
    except Exception as shadow_error:
        shadow_result = {'status': 'error', 'model_version': SHADOW_MODEL_VERSION, 'error': f'{type(shadow_error).__name__}: {shadow_error}', 'all_combo_predictions': []}
    combined_race_html = racelist_html + before_html
    race_grade = resolve_race_grade(target_date, venue_code, combined_race_html)
    page_text = BeautifulSoup(combined_race_html, 'html.parser').get_text(' ', strip=True)
    special_keywords = ['安定板', '1200m', '進入固定', '周回短縮', '展示航走中止', 'レース中止']
    specials = [x for x in special_keywords if x in page_text]
    rec = out[(out['safe_probability'] >= 0.008) & (out['safe_EV'] >= 1.15) & (out['odds'] <= 150) & (out['market_disagreement'] <= 2.5)].copy()
    if specials:
        rec = rec[(rec['safe_EV'] >= 1.25) & (rec['odds'] <= 100)]
    rec = rec.sort_values(['safe_EV', 'safe_probability'], ascending=False).head(6)
    recommendation_rows = [{'combo': r.combo, 'safe_probability': round(float(r.safe_probability), 6), 'odds': round(float(r.odds), 1), 'safe_EV': round(float(r.safe_EV), 3), 'market_disagreement': round(float(r.market_disagreement), 3)} for r in rec.itertuples()]
    recommendation_rows = allocate_stakes_by_odds(recommendation_rows, budget=1000, unit=100, target_payout=20000, min_stake=100)
    canonical_race_name = f'{VENUES[venue_code]}{race_no}R'
    return {'race': canonical_race_name, 'date': target_date.isoformat(), 'venue_code': venue_code, 'race_no': race_no, 'race_grade': race_grade, 'special_conditions': specials, 'fetched_at': datetime.now(JST).isoformat(), 'budget': 1000, 'race_hit_probability': round(min(1.0, sum((float(r.get('safe_probability') or 0) for r in recommendation_rows))), 6), 'recommendations': recommendation_rows, 'shadow_model': shadow_result, 'all_combo_predictions': [{'combo': r.combo, 'safe_probability': round(float(r.safe_probability), 6), 'odds': round(float(r.odds), 1), 'safe_EV': round(float(r.safe_EV), 3), 'rank': int(rank)} for rank, r in enumerate(out.sort_values(['safe_probability', 'safe_EV'], ascending=False).itertuples(), start=1)], 'top10': [{'combo': r.combo, 'probability': round(float(r.probability), 6)} for r in out.sort_values('probability', ascending=False).head(10).itertuples()]}
_firestore_client = None

def get_firestore():
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore.Client()
    return _firestore_client

def prediction_doc_id(result):
    return f"{result['date']}_{int(result['venue_code']):02d}_{int(result['race_no']):02d}"

def save_prediction_log(result, source='manual'):
    db = get_firestore()
    recommendations = result.get('recommendations', [])
    now = datetime.now(JST)
    payload = {'source': source, 'race': result['race'], 'date': result['date'], 'venue_code': int(result['venue_code']), 'race_no': int(result['race_no']), 'race_grade': result.get('race_grade', '一般'), 'grade_detection_version': 5, 'special_conditions': result.get('special_conditions', []), 'fetched_at': result.get('fetched_at'), 'created_at': now, 'decision': '推奨' if recommendations else '見送り', 'race_hit_probability': result.get('race_hit_probability', 0), 'recommendations': recommendations, 'all_combo_predictions': result.get('all_combo_predictions', []), 'shadow_model': result.get('shadow_model', {'status': 'unavailable', 'model_version': SHADOW_MODEL_VERSION, 'all_combo_predictions': []}), 'result_checked': False, 'result_combo': None, 'trifecta_payout': None, 'stake': sum((int(r.get('stake') or 0) for r in recommendations)), 'payout': None, 'profit': None, 'hit': None}
    doc_id = prediction_doc_id(result)
    db.collection('predictions').document(doc_id).set(payload)
    return doc_id

def parse_result_combo(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(' ', strip=True)
    for tr in soup.find_all('tr'):
        row_text = ' '.join(tr.stripped_strings)
        if '3連単' not in row_text and '三連単' not in row_text:
            continue
        m = re.search('([1-6])\\s*[-－]\\s*([1-6])\\s*[-－]\\s*([1-6])', row_text)
        if m and len(set(m.groups())) == 3:
            return '-'.join(m.groups())
    combos = re.findall('([1-6])\\s*[-－]\\s*([1-6])\\s*[-－]\\s*([1-6])', text)
    for combo in combos:
        if len(set(combo)) == 3:
            return '-'.join(combo)
    raise RuntimeError('result not published yet')

def parse_trifecta_payout(html, combo):
    """
    公式結果ページの払戻表から、
    3連単の100円あたり払戻金だけを厳密に取得する。
    """
    target_combo = str(combo).replace('－', '-').replace('–', '-').replace('—', '-').replace(' ', '')

    def normalize_combo(value):
        return str(value).replace('－', '-').replace('–', '-').replace('—', '-').replace(' ', '')

    def extract_money(value):
        value = str(value).strip()
        match = re.search('(?:[¥￥]\\s*([\\d,]+)|([\\d,]+)\\s*円)', value)
        if not match:
            return None
        raw = match.group(1) or match.group(2)
        amount = int(raw.replace(',', ''))
        return amount if amount >= 100 else None
    soup = BeautifulSoup(html, 'html.parser')
    for tr in soup.find_all('tr'):
        cells = [cell.get_text(' ', strip=True) for cell in tr.find_all(['th', 'td'])]
        if not cells:
            continue
        row_text = ' '.join(cells)
        if '3連単' not in row_text and '三連単' not in row_text:
            continue
        combo_found = any((normalize_combo(cell) == target_combo for cell in cells))
        if not combo_found:
            continue
        for cell in cells:
            amount = extract_money(cell)
            if amount is not None:
                return amount
    try:
        tables = pd.read_html(StringIO(html), header=None)
        for table in tables:
            for _, row in table.iterrows():
                cells = [str(value).strip() for value in row.tolist() if not pd.isna(value)]
                row_text = ' '.join(cells)
                if '3連単' not in row_text and '三連単' not in row_text:
                    continue
                combo_found = any((normalize_combo(cell) == target_combo for cell in cells))
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
    hd = str(date_text).replace('-', '')
    result_html, url = get_html('raceresult', hd, int(venue_code), int(race_no))
    try:
        racelist_html, _ = get_html('racelist', hd, int(venue_code), int(race_no))
    except Exception:
        racelist_html = ''
    combo = parse_result_combo(result_html)
    payout = parse_trifecta_payout(result_html, combo)
    result_date = datetime.strptime(str(date_text)[:10], '%Y-%m-%d').date()
    grade = resolve_race_grade(result_date, int(venue_code), result_html + racelist_html)
    return (combo, payout, url, grade)

def update_pending_results(limit=100):
    db = get_firestore()
    docs = db.collection('predictions').limit(max(limit, 200)).stream()
    checked = 0
    updated = 0
    pending = 0
    errors = []
    for doc in docs:
        data = doc.to_dict()
        needs_check = data.get('result_checked') is not True
        needs_payout_recheck = data.get('hit') is True
        needs_grade_backfill = int(data.get('grade_detection_version') or 0) < 5
        if not needs_check and (not needs_payout_recheck) and (not needs_grade_backfill):
            continue
        checked += 1
        try:
            combo, trifecta_payout, _, race_grade = fetch_official_result(data['date'], data['venue_code'], data['race_no'])
            recommendations = data.get('recommendations', [])
            combos = [str(x.get('combo', '')) for x in recommendations]
            hit = combo in combos
            all_combo_predictions = data.get('all_combo_predictions', [])
            result_prediction = next((item for item in all_combo_predictions if str(item.get('combo', '')) == combo), None)
            if result_prediction is None:
                result_prediction = next((item for item in recommendations if str(item.get('combo', '')) == combo), None)
            result_predicted_probability = None
            result_predicted_odds = None
            result_predicted_ev = None
            result_predicted_rank = None
            shadow_model = data.get('shadow_model') or {}
            shadow_predictions = shadow_model.get('all_combo_predictions', []) or []
            shadow_result_prediction = next((item for item in shadow_predictions if str(item.get('combo', '')) == combo), None)
            shadow_result_probability = None
            shadow_result_odds = None
            shadow_result_ev = None
            shadow_result_rank = None
            shadow_top1_hit = False
            shadow_top3_hit = False
            shadow_top5_hit = False
            shadow_top10_hit = False
            shadow_top20_hit = False
            if shadow_result_prediction:
                shadow_result_probability = shadow_result_prediction.get('probability')
                shadow_result_odds = shadow_result_prediction.get('odds')
                shadow_result_ev = shadow_result_prediction.get('EV')
                shadow_result_rank = shadow_result_prediction.get('rank')
                try:
                    sr = int(shadow_result_rank)
                    shadow_top1_hit = sr <= 1
                    shadow_top3_hit = sr <= 3
                    shadow_top5_hit = sr <= 5
                    shadow_top10_hit = sr <= 10
                    shadow_top20_hit = sr <= 20
                except (TypeError, ValueError):
                    pass
            if result_prediction:
                result_predicted_probability = result_prediction.get('safe_probability')
                result_predicted_odds = result_prediction.get('odds')
                result_predicted_ev = result_prediction.get('safe_EV')
                result_predicted_rank = result_prediction.get('rank')
            stake = int(data.get('stake') or sum((int(x.get('stake') or 0) for x in recommendations)))
            winning_stake = 0
            for item in recommendations:
                if str(item.get('combo', '')) == combo:
                    winning_stake = int(item.get('stake') or 0)
                    break
            payout = 0
            if hit and trifecta_payout is not None and (winning_stake > 0):
                payout = int(round(float(trifecta_payout) * winning_stake / 100))
            if hit and trifecta_payout is None:
                pending += 1
                continue
            profit = payout - stake
            doc.reference.update({'result_checked': True, 'result_checked_at': datetime.now(JST), 'result_combo': combo, 'race_grade': race_grade, 'grade_detection_version': 5, 'result_predicted_probability': result_predicted_probability, 'result_predicted_odds': result_predicted_odds, 'result_predicted_ev': result_predicted_ev, 'result_predicted_rank': result_predicted_rank, 'result_was_recommended': hit, 'shadow_result_probability': shadow_result_probability, 'shadow_result_odds': shadow_result_odds, 'shadow_result_ev': shadow_result_ev, 'shadow_result_rank': shadow_result_rank, 'shadow_top1_hit': shadow_top1_hit, 'shadow_top3_hit': shadow_top3_hit, 'shadow_top5_hit': shadow_top5_hit, 'shadow_top10_hit': shadow_top10_hit, 'shadow_top20_hit': shadow_top20_hit, 'shadow_model_version': shadow_model.get('model_version', SHADOW_MODEL_VERSION), 'shadow_model_status': shadow_model.get('status', 'unavailable'), 'trifecta_payout': trifecta_payout, 'hit': hit, 'payout': payout, 'profit': profit})
            updated += 1
        except RuntimeError:
            pending += 1
        except Exception as e:
            errors.append({'doc_id': doc.id, 'error': f'{type(e).__name__}: {e}'})
    return {'checked': checked, 'updated': updated, 'pending': pending, 'errors': errors[:10]}

def delete_all_predictions():
    db = get_firestore()
    deleted = 0
    while True:
        docs = list(db.collection('predictions').limit(200).stream())
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
    docs = list(db.collection('predictions').order_by('created_at', direction=firestore.Query.DESCENDING).limit(1000).stream())
    raw_rows = [doc.to_dict() | {'id': doc.id} for doc in docs]
    latest_by_race = {}
    for row in raw_rows:
        key = (str(row.get('date')), int(row.get('venue_code') or 0), int(row.get('race_no') or 0))
        current = latest_by_race.get(key)
        if current is None:
            latest_by_race[key] = row
            continue
        row_time = row.get('created_at')
        current_time = current.get('created_at')
        if current_time is None or (row_time is not None and row_time > current_time):
            latest_by_race[key] = row
    rows = sorted(latest_by_race.values(), key=lambda r: r.get('created_at') or datetime.min.replace(tzinfo=JST), reverse=True)

    def normalized_grade(row):
        grade = str(row.get('race_grade') or '一般').upper()
        if grade == 'SG':
            return 'SG'
        if grade in {'G1', 'GⅠ', 'ＧⅠ'}:
            return 'G1'
        if grade in {'G2', 'GⅡ', 'ＧⅡ'}:
            return 'G2'
        if grade in {'G3', 'GⅢ', 'ＧⅢ'}:
            return 'G3'
        return '一般'

    def summarize(target_rows):
        finished = [r for r in target_rows if r.get('result_checked') is True]
        recommended = [r for r in finished if r.get('decision') == '推奨']
        total_stake = sum((int(r.get('stake') or 0) for r in recommended))
        total_payout = sum((int(r.get('payout') or 0) for r in recommended))
        total_profit = sum((int(r.get('profit') or 0) for r in recommended))
        hit_races = sum((1 for r in recommended if r.get('hit') is True))
        venue_map = {}
        for row in recommended:
            venue_code = row.get('venue_code')
            try:
                venue_code = int(venue_code)
            except (TypeError, ValueError):
                venue_code = None
            if venue_code in VENUES:
                venue = VENUES[venue_code]
            else:
                race_name = str(row.get('race') or '')
                venue = re.sub('\\d{1,2}R?$', '', race_name, flags=re.I) or '不明'
            item = venue_map.setdefault(venue, {'venue': venue, 'predictions': 0, 'hits': 0, 'stake': 0, 'payout': 0, 'probability_sum': 0.0, 'probability_count': 0})
            item['predictions'] += 1
            item['hits'] += 1 if row.get('hit') is True else 0
            item['stake'] += int(row.get('stake') or 0)
            item['payout'] += int(row.get('payout') or 0)
            probability = row.get('race_hit_probability')
            if probability is None:
                probability = min(1.0, sum((float(x.get('safe_probability') or 0) for x in row.get('recommendations', []))))
            item['probability_sum'] += float(probability or 0)
            item['probability_count'] += 1
        venue_stats = []
        for item in venue_map.values():
            count = item['predictions']
            stake = item['stake']
            payout = item['payout']
            probability_count = item['probability_count']
            venue_stats.append({'venue': item['venue'], 'predictions': count, 'hits': item['hits'], 'stake': stake, 'payout': payout, 'hit_rate': item['hits'] / count if count else None, 'average_race_hit_probability': item['probability_sum'] / probability_count if probability_count else None, 'profit': payout - stake, 'roi': payout / stake if stake else None})
        venue_stats.sort(key=lambda x: (x['predictions'], x['hit_rate'] or 0), reverse=True)
        recent = []
        for r in target_rows[:30]:
            recent.append({'id': r.get('id'), 'source': r.get('source', 'manual'), 'race': r.get('race'), 'race_grade': normalized_grade(r), 'date': r.get('date'), 'fetched_at': r.get('fetched_at'), 'decision': r.get('decision'), 'race_hit_probability': r.get('race_hit_probability', min(1.0, sum((float(x.get('safe_probability') or 0) for x in r.get('recommendations', []))))), 'recommendations': r.get('recommendations', []), 'special_conditions': r.get('special_conditions', []), 'result_combo': r.get('result_combo'), 'result_predicted_probability': r.get('result_predicted_probability'), 'result_predicted_odds': r.get('result_predicted_odds'), 'result_predicted_ev': r.get('result_predicted_ev'), 'result_predicted_rank': r.get('result_predicted_rank'), 'result_was_recommended': r.get('result_was_recommended', r.get('hit')), 'hit': r.get('hit'), 'stake': r.get('stake'), 'payout': r.get('payout'), 'profit': r.get('profit')})
        return {'saved_predictions': len(target_rows), 'finished_races': len(finished), 'recommended_finished': len(recommended), 'hit_races': hit_races, 'hit_rate': hit_races / len(recommended) if recommended else None, 'total_stake': total_stake, 'total_payout': total_payout, 'total_profit': total_profit, 'roi': total_payout / total_stake if total_stake else None, 'venue_stats': venue_stats, 'recent': recent}
    comparable_rows = [row for row in rows if row.get('result_checked') is True and row.get('result_predicted_rank') is not None and (row.get('shadow_result_rank') is not None) and (row.get('shadow_model_status') == 'ok')]

    def topn_rate(target_rows, field, n):
        ranks = []
        for row in target_rows:
            try:
                ranks.append(int(row.get(field)))
            except (TypeError, ValueError):
                continue
        if not ranks:
            return None
        return sum((1 for rank in ranks if rank <= n)) / len(ranks)
    comparison = {'races': len(comparable_rows), 'old': {'top1': topn_rate(comparable_rows, 'result_predicted_rank', 1), 'top3': topn_rate(comparable_rows, 'result_predicted_rank', 3), 'top5': topn_rate(comparable_rows, 'result_predicted_rank', 5), 'top10': topn_rate(comparable_rows, 'result_predicted_rank', 10), 'top20': topn_rate(comparable_rows, 'result_predicted_rank', 20)}, 'new': {'top1': topn_rate(comparable_rows, 'shadow_result_rank', 1), 'top3': topn_rate(comparable_rows, 'shadow_result_rank', 3), 'top5': topn_rate(comparable_rows, 'shadow_result_rank', 5), 'top10': topn_rate(comparable_rows, 'shadow_result_rank', 10), 'top20': topn_rate(comparable_rows, 'shadow_result_rank', 20)}}

    def normalize_combo(value):
        text = str(value or '').strip()
        digits = [char for char in text if char.isdigit()]
        if len(digits) >= 3:
            return '-'.join(digits[:3])
        return text

    def get_prediction_items(row, model_name):
        if model_name == 'new':
            shadow = row.get('shadow_model') or {}
            return shadow.get('all_combo_predictions') or []
        for field in ('all_combo_predictions', 'predictions', 'combo_predictions'):
            items = row.get(field)
            if isinstance(items, list) and items:
                return items
        return []

    def get_item_ev(item, model_name):
        fields = ('EV', 'ev') if model_name == 'new' else ('safe_EV', 'safe_ev', 'EV', 'ev')
        for field in fields:
            try:
                value = item.get(field)
                if value is not None:
                    return float(value)
            except (TypeError, ValueError, AttributeError):
                continue
        return None

    def get_result_combo(row):
        for field in ('result_combo', 'actual_combo', 'result_trifecta', 'trifecta_result'):
            value = row.get(field)
            if value:
                return normalize_combo(value)
        return ''

    def get_result_payout(row):
        for field in ('trifecta_payout', 'result_payout', 'payout', 'result_trifecta_payout'):
            try:
                value = row.get(field)
                if value is not None:
                    return int(float(value))
            except (TypeError, ValueError):
                continue
        return 0

    def virtual_roi(target_rows, model_name, min_ev=1.0, max_picks=6, stake_per_pick=100):
        total_stake = 0
        total_payout = 0
        bet_races = 0
        hit_races = 0
        insufficient_snapshot_races = 0
        unstable_picks_excluded = 0

        def snapshot_map(snapshot):
            items = snapshot.get('all_combo_predictions') or []
            result = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                combo = normalize_combo(item.get('combo'))
                if combo:
                    result[combo] = item
            return result
        for row in target_rows:
            result_combo = get_result_combo(row)
            payout = get_result_payout(row)
            if not result_combo or payout <= 0:
                continue
            candidates = []
            if model_name == 'new':
                snapshots = row.get('shadow_odds_snapshots') or []
                if not isinstance(snapshots, list) or len(snapshots) < 2:
                    insufficient_snapshot_races += 1
                    continue
                previous_map = snapshot_map(snapshots[-2])
                latest_map = snapshot_map(snapshots[-1])
                for combo in sorted(set(previous_map) & set(latest_map)):
                    previous_item = previous_map[combo]
                    latest_item = latest_map[combo]
                    try:
                        previous_odds = float(previous_item.get('odds'))
                        latest_odds = float(latest_item.get('odds'))
                        probability = float(latest_item.get('probability'))
                    except (TypeError, ValueError):
                        continue
                    if previous_odds <= 0 or latest_odds <= 0 or probability <= 0:
                        continue
                    lower_odds = min(previous_odds, latest_odds)
                    upper_odds = max(previous_odds, latest_odds)
                    odds_ratio = upper_odds / lower_odds
                    if odds_ratio > 2.0:
                        unstable_picks_excluded += 1
                        continue
                    safe_ev = probability * lower_odds
                    if safe_ev >= min_ev:
                        candidates.append((safe_ev, combo))
            else:
                for item in get_prediction_items(row, model_name):
                    if not isinstance(item, dict):
                        continue
                    ev = get_item_ev(item, model_name)
                    if ev is None or ev < min_ev:
                        continue
                    combo = normalize_combo(item.get('combo'))
                    if combo:
                        candidates.append((ev, combo))
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            candidates = candidates[:max_picks]
            if not candidates:
                continue
            bet_races += 1
            total_stake += len(candidates) * stake_per_pick
            picked = {combo for _, combo in candidates}
            if result_combo in picked:
                hit_races += 1
                total_payout += payout
        return {'bet_races': bet_races, 'hit_races': hit_races, 'stake': total_stake, 'payout': total_payout, 'profit': total_payout - total_stake, 'roi': total_payout / total_stake if total_stake else None, 'insufficient_snapshot_races': insufficient_snapshot_races, 'unstable_picks_excluded': unstable_picks_excluded, 'safe_odds_rule': 'latest_two_lower_odds', 'max_odds_ratio': 2.0}
    comparison['virtual_rule'] = {'min_ev': 1.0, 'max_picks': 6, 'stake_per_pick': 100}
    comparison['old_virtual'] = virtual_roi(comparable_rows, 'old')
    comparison['new_virtual'] = virtual_roi(comparable_rows, 'new')
    groups = {'all': rows, 'SG': [r for r in rows if normalized_grade(r) == 'SG'], 'G1': [r for r in rows if normalized_grade(r) == 'G1'], 'G2': [r for r in rows if normalized_grade(r) == 'G2'], 'G3': [r for r in rows if normalized_grade(r) == 'G3'], '一般': [r for r in rows if normalized_grade(r) == '一般']}
    shadow_history = []
    for row in comparable_rows:
        shadow = row.get('shadow_model') or {}
        items = shadow.get('all_combo_predictions') or []
        picks = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                ev = float(item.get('EV'))
            except (TypeError, ValueError):
                continue
            if ev < 1.0:
                continue
            picks.append({'combo': item.get('combo'), 'probability': item.get('probability'), 'odds': item.get('odds'), 'EV': ev, 'rank': item.get('rank')})
        picks.sort(key=lambda item: item.get('EV') or 0, reverse=True)
        picks = picks[:6]
        result_combo = row.get('result_combo') or row.get('actual_combo') or row.get('result_trifecta') or ''
        payout = row.get('trifecta_payout') or row.get('result_payout') or row.get('payout') or 0
        picked = {str(item.get('combo') or '') for item in picks}
        shadow_history.append({'date': row.get('date') or row.get('race_date'), 'venue': row.get('venue') or row.get('venue_name'), 'race_no': row.get('race_no'), 'fetched_at': shadow.get('fetched_at') or row.get('shadow_fetched_at'), 'minutes_before_deadline': shadow.get('minutes_before_deadline') if shadow.get('minutes_before_deadline') is not None else row.get('shadow_minutes_before_deadline'), 'near_deadline_refresh': bool(shadow.get('near_deadline_refresh')), 'picks': picks, 'result_combo': result_combo, 'payout': payout, 'hit': str(result_combo) in picked})
    shadow_history.sort(key=lambda item: str(item.get('fetched_at') or ''), reverse=True)
    shadow_history = shadow_history[:50]
    return {'shadow_history': shadow_history, 'comparison': comparison, 'groups': {key: summarize(group_rows) for key, group_rows in groups.items()}}
HTML_PAGE = '\n<!doctype html>\n<html lang="ja">\n<head>\n  <meta charset="utf-8">\n  <meta name="viewport" content="width=device-width,initial-scale=1">\n  <title>競艇AI</title>\n  <style>\n    :root {\n      color-scheme: dark;\n      --bg: #0e1116;\n      --panel: #171b22;\n      --line: #2a303a;\n      --text: #f4f6f8;\n      --muted: #a9b0bb;\n      --accent: #5ca7ff;\n      --good: #53d18b;\n      --warn: #ffca5c;\n    }\n    * { box-sizing: border-box; }\n    body {\n      margin: 0;\n      background: var(--bg);\n      color: var(--text);\n      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",\n                   "Noto Sans JP", sans-serif;\n    }\n    main {\n      width: min(720px, 100%);\n      margin: 0 auto;\n      padding: 22px 16px 60px;\n    }\n    h1 { margin: 4px 0 6px; font-size: 28px; }\n    .sub { color: var(--muted); margin-bottom: 22px; }\n    form { display: flex; gap: 10px; margin-bottom: 18px; }\n    input {\n      min-width: 0;\n      flex: 1;\n      font-size: 18px;\n      padding: 14px 15px;\n      border-radius: 12px;\n      border: 1px solid var(--line);\n      background: var(--panel);\n      color: var(--text);\n    }\n    button {\n      border: 0;\n      border-radius: 12px;\n      padding: 0 18px;\n      font-size: 16px;\n      font-weight: 700;\n      color: white;\n      background: var(--accent);\n    }\n    .status {\n      color: var(--muted);\n      min-height: 24px;\n      margin: 8px 0 16px;\n    }\n    .meta, .card, .empty {\n      background: var(--panel);\n      border: 1px solid var(--line);\n      border-radius: 14px;\n      padding: 15px;\n      margin-bottom: 12px;\n    }\n    .meta { line-height: 1.7; }\n    .special { color: var(--warn); font-weight: 700; }\n    .combo { font-size: 25px; font-weight: 800; margin-bottom: 8px; }\n    .grid {\n      display: grid;\n      grid-template-columns: 1fr 1fr;\n      gap: 7px 14px;\n      color: var(--muted);\n    }\n    .grid b { color: var(--text); }\n    .ev { color: var(--good) !important; }\n    .empty { text-align: center; font-size: 20px; padding: 30px 15px; }\n    .secondary { background: #343b46; }\n    .danger { background: #c94a4a; }\n    .stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }\n    .stat { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; }\n    .stat small { display:block; color:var(--muted); margin-bottom:5px; }\n    .stat strong { font-size:22px; }\n    .history { margin-top:14px; }\n    .history-row {\n      border-bottom:1px solid var(--line);\n      padding:12px 0;\n      font-size:14px;\n      cursor:pointer;\n    }\n    .history-row:last-child { border-bottom:0; }\n    .history-summary {\n      display:flex;\n      justify-content:space-between;\n      gap:10px;\n      align-items:center;\n    }\n    .history-arrow { color:var(--muted); font-size:18px; }\n    .history-detail {\n      display:none;\n      margin-top:10px;\n      padding:12px;\n      border-radius:10px;\n      background:#10141a;\n      border:1px solid var(--line);\n    }\n    .history-row.open .history-detail { display:block; }\n    .history-pick {\n      display:grid;\n      grid-template-columns:1fr auto auto;\n      gap:8px;\n      padding:7px 0;\n      border-bottom:1px solid var(--line);\n    }\n    .history-pick:last-child { border-bottom:0; }\n    .history-muted { color:var(--muted); font-size:12px; }\n    .foot {\n      color: var(--muted);\n      font-size: 13px;\n      line-height: 1.6;\n      margin-top: 18px;\n    }\n  \n    /* Mobile layout */\n    @media (max-width: 640px) {\n      .form-row,\n      form {\n        display: grid !important;\n        grid-template-columns: 1fr 1fr !important;\n        gap: 10px !important;\n        align-items: stretch !important;\n      }\n\n      input[type="text"],\n      input[type="search"],\n      #raceInput {\n        grid-column: 1 / -1 !important;\n        width: 100% !important;\n        min-width: 0 !important;\n        height: 56px !important;\n        box-sizing: border-box !important;\n        font-size: 16px !important;\n      }\n\n      button {\n        width: 100% !important;\n        min-width: 0 !important;\n        min-height: 56px !important;\n        padding: 10px 8px !important;\n        font-size: 16px !important;\n        line-height: 1.25 !important;\n        white-space: nowrap !important;\n      }\n\n      #refreshBtn {\n        font-size: 15px !important;\n      }\n\n      .container,\n      main {\n        width: 100% !important;\n        max-width: 100% !important;\n        padding-left: 16px !important;\n        padding-right: 16px !important;\n        box-sizing: border-box !important;\n      }\n\n      .stats-grid {\n        gap: 10px !important;\n      }\n\n      .card,\n      .stat-card {\n        min-width: 0 !important;\n      }\n    }\n\n    \n    @media (max-width: 640px) {\n      .grade-tabs {\n        grid-template-columns: repeat(3, 1fr);\n      }\n    }\n\n@media (max-width: 390px) {\n      button {\n        font-size: 15px !important;\n        padding-left: 6px !important;\n        padding-right: 6px !important;\n      }\n\n      #refreshBtn {\n        font-size: 14px !important;\n      }\n    }\n\n\n    .grade-tabs {\n      display: grid;\n      grid-template-columns: repeat(6, 1fr);\n      gap: 7px;\n      margin: 14px 0;\n    }\n    .grade-tab {\n      min-height: 44px;\n      padding: 8px 4px;\n      border: 1px solid var(--line);\n      background: var(--panel);\n      color: var(--muted);\n      border-radius: 11px;\n      font-size: 14px;\n      white-space: nowrap;\n    }\n    .grade-tab.active {\n      background: var(--accent);\n      color: white;\n      border-color: var(--accent);\n    }\n    @media (max-width: 390px) {\n      .grade-tabs { gap: 5px; }\n      .grade-tab {\n        font-size: 12px !important;\n        padding: 7px 2px !important;\n      }\n    }\n\n</style>\n</head>\n<body>\n<main>\n  <h1>🚤 競艇AI</h1>\n  <div class="sub">レース名を入れるだけで保守的候補を表示</div>\n\n  <form id="form">\n    <input id="race" value="蒲郡12R" placeholder="例：浜名湖6R">\n    <button type="submit">予想</button>\n    <button type="button" id="statsBtn" class="secondary">成績</button>\n    <button type="button" id="refreshBtn" class="secondary">結果を更新</button>\n    <button type="button" id="deleteBtn" class="danger">全削除</button>\n  </form>\n\n  <div id="status" class="status"></div>\n  <div id="result"></div>\n\n  <div class="foot">\n    生AI確率ではなく、補正確率・保守EV・市場乖離を使って候補を絞っています。<br>\n    候補がない場合は「見送り」が正常です。\n  </div>\n</main>\n\n<script>\nconst form = document.getElementById("form");\nconst input = document.getElementById("race");\nconst statusBox = document.getElementById("status");\nconst resultBox = document.getElementById("result");\nconst statsBtn = document.getElementById("statsBtn");\nconst refreshBtn = document.getElementById("refreshBtn");\nconst deleteBtn = document.getElementById("deleteBtn");\n\nfunction esc(v) {\n  return String(v ?? "")\n    .replaceAll("&", "&amp;")\n    .replaceAll("<", "&lt;")\n    .replaceAll(">", "&gt;")\n    .replaceAll(\'"\', "&quot;");\n}\n\nform.addEventListener("submit", async (e) => {\n  e.preventDefault();\n  const race = input.value.trim();\n  if (!race) return;\n\n  statusBox.textContent = "予想中… 初回は少し時間がかかります";\n  resultBox.innerHTML = "";\n\n  try {\n    const res = await fetch("/predict?race=" + encodeURIComponent(race));\n    const data = await res.json();\n\n    if (!res.ok) {\n      throw new Error(data.message || data.error || "予想に失敗しました");\n    }\n\n    const specials = data.special_conditions || [];\n    let html = `\n      <div class="meta">\n        <b>${esc(data.race)}</b><br>\n        日付：${esc(data.date)}<br>\n        取得：${esc(data.fetched_at)}<br>\n        投資上限：${Number(data.budget || 0).toLocaleString()}円<br>\n        実際の投資額：${(data.recommendations || []).reduce(\n          (sum, r) => sum + Number(r.stake || 0),\n          0\n        ).toLocaleString()}円<br>\n        配分方式：${\n          (data.recommendations || []).length <= 4\n            ? "払戻20,000円以上になる最小額"\n            : "全候補へ合計1,000円を配分"\n        }<br>\n        レース的中見込み：\n        <b>${(Number(data.race_hit_probability || 0) * 100).toFixed(1)}%</b>\n        ${\n          specials.length\n          ? `<br><span class="special">⚠️ ${esc(specials.join(" / "))}</span>`\n          : ""\n        }\n      </div>\n    `;\n\n    const recs = data.recommendations || [];\n\n    if (recs.length === 0) {\n      html += `<div class="empty">今回は見送り</div>`;\n    } else {\n      recs.forEach((r, i) => {\n        html += `\n          <div class="card">\n            <div class="combo">${i + 1}位\u3000${esc(r.combo)}</div>\n            <div class="grid">\n              <span>補正確率</span>\n              <b>${(Number(r.safe_probability) * 100).toFixed(2)}%</b>\n              <span>現在オッズ</span>\n              <b>${Number(r.odds).toFixed(1)}倍</b>\n              <span>保守EV</span>\n              <b class="ev">${Number(r.safe_EV).toFixed(2)}</b>\n              <span>市場乖離</span>\n              <b>${Number(r.market_disagreement).toFixed(2)}倍</b>\n              <span>購入額</span>\n              <b>${Number(r.stake || 0).toLocaleString()}円</b>\n              <span>的中時払戻</span>\n              <b>${Number(r.gross_if_hit || 0).toLocaleString()}円</b>\n              <span>的中時収支</span>\n              <b class="ev">${Number(r.net_if_hit || 0).toLocaleString()}円</b>\n            </div>\n          </div>\n        `;\n      });\n    }\n\n    resultBox.innerHTML = html;\n    statusBox.textContent = data.log_id ? "予想完了・自動保存済み" : "予想完了";\n  } catch (err) {\n    statusBox.textContent = "エラー";\n    resultBox.innerHTML = `<div class="empty">${esc(err.message)}</div>`;\n  }\n});\n\nstatsBtn.addEventListener("click", async () => {\n  statusBox.textContent = "成績を読み込み中…";\n  resultBox.innerHTML = "";\n\n  try {\n    const res = await fetch("/stats");\n    const payload = await res.json();\n\n    if (!res.ok) {\n      throw new Error(\n        payload.message\n        || payload.error\n        || "成績取得失敗"\n      );\n    }\n\n    const groups = payload.groups || {};\n    const all = groups.all || {};\n    const comparison = payload.comparison || {};\n    const shadowHistory = payload.shadow_history || [];\n\n    const pct = (v) =>\n      v == null\n        ? "-"\n        : (Number(v) * 100).toFixed(1) + "%";\n\n    const yen = (v) =>\n      Number(v || 0).toLocaleString() + "円";\n\n    const labels = {\n      all: "全て",\n      SG: "SG",\n      G1: "G1",\n      G2: "G2",\n      G3: "G3",\n      "一般": "一般"\n    };\n\n    const renderGradeSection = (grade) => {\n      const s = groups[grade] || {\n        venue_stats: [],\n        recent: []\n      };\n\n      const venues = s.venue_stats || [];\n\n      const venueHtml = venues.length === 0\n        ? `<div class="empty">\n            確定済みの推奨予想がまだありません\n          </div>`\n        : venues.map((v) => `\n            <div class="history-row">\n              <div class="history-summary">\n                <div>\n                  <b>${esc(v.venue)}</b><br>\n                  ${Number(v.hits || 0)}\n                  / ${Number(v.predictions || 0)}的中\n                  ・的中率 ${pct(v.hit_rate)}<br>\n                  平均レース的中見込み\n                  ${pct(v.average_race_hit_probability)}\n                </div>\n                <div style="text-align:right">\n                  回収率 ${pct(v.roi)}<br>\n                  ${yen(v.profit)}\n                </div>\n              </div>\n            </div>\n          `).join("");\n\n      let recentHtml = "";\n\n      (s.recent || []).forEach((r, index) => {\n        const hit = r.hit === true\n          ? "🎯 的中"\n          : (\n              r.hit === false\n                ? "はずれ"\n                : "未確定"\n            );\n\n        const recs = r.recommendations || [];\n        let picksHtml = "";\n\n        if (recs.length === 0) {\n          picksHtml = `\n            <div class="history-muted">\n              この予想は見送りでした\n            </div>\n          `;\n        } else {\n          recs.forEach((pick, i) => {\n            picksHtml += `\n              <div class="history-pick">\n                <b>${i + 1}位 ${esc(pick.combo)}</b>\n                <span>\n                  ${Number(pick.odds || 0).toFixed(1)}倍\n                </span>\n                <span>\n                  ${Number(pick.stake || 0).toLocaleString()}円\n                </span>\n              </div>\n              <div class="history-muted">\n                補正確率\n                ${(Number(pick.safe_probability || 0) * 100).toFixed(2)}%\n                / 保守EV\n                ${Number(pick.safe_EV || 0).toFixed(2)}\n              </div>\n            `;\n          });\n        }\n\n        const conditions =\n          (r.special_conditions || []).length\n            ? `<div class="special">\n                ⚠️ ${esc(r.special_conditions.join(" / "))}\n              </div>`\n            : "";\n\n        recentHtml += `\n          <div\n            class="history-row"\n            data-history-index="${index}"\n          >\n            <div class="history-summary">\n              <div>\n                <b>\n                  ${esc(r.date)} ${esc(r.race)}\n                  ${r.source === "auto" ? " 🤖" : ""}\n                </b><br>\n                ${esc(r.decision)} / ${hit}\n                ${\n                  r.profit == null\n                    ? ""\n                    : ` / ${yen(r.profit)}`\n                }\n              </div>\n              <span class="history-arrow">›</span>\n            </div>\n\n            <div class="history-detail">\n              <div class="history-muted">\n                グレード：${esc(r.race_grade || "一般")}<br>\n                予想時刻：${esc(r.fetched_at || "-")}<br>\n                レース的中見込み：\n                <b>\n                  ${(Number(r.race_hit_probability || 0) * 100).toFixed(1)}%\n                </b>\n              </div>\n\n              ${conditions}\n\n              <div style="margin-top:8px">\n                ${picksHtml}\n              </div>\n\n              ${\n                r.result_combo\n                  ? `<div style="margin-top:10px">\n                      結果：\n                      <b>${esc(r.result_combo)}</b>\n                      / 払戻 ${yen(r.payout)}\n\n                      ${\n                        r.result_predicted_probability == null\n                          ? `<div class="history-muted" style="margin-top:8px">\n                              結果着順の事前予想値：\n                              旧形式データのため保存なし\n                            </div>`\n                          : `<div class="history-detail"\n                                  style="display:block;margin-top:8px">\n                              <b>結果着順の事前評価</b><br>\n                              補正確率：\n                              ${(Number(\n                                r.result_predicted_probability\n                              ) * 100).toFixed(2)}%<br>\n                              予想時オッズ：\n                              ${Number(\n                                r.result_predicted_odds || 0\n                              ).toFixed(1)}倍<br>\n                              保守EV：\n                              ${Number(\n                                r.result_predicted_ev || 0\n                              ).toFixed(2)}<br>\n                              予想順位：\n                              ${\n                                r.result_predicted_rank == null\n                                  ? "-"\n                                  : `${Number(\n                                      r.result_predicted_rank\n                                    )}位 / 120通り`\n                              }<br>\n                              推奨対象：\n                              ${\n                                r.result_was_recommended\n                                  ? "入っていた"\n                                  : "入っていなかった"\n                              }\n                            </div>`\n                      }\n                    </div>`\n                  : ""\n              }\n            </div>\n          </div>\n        `;\n      });\n\n      if (!recentHtml) {\n        recentHtml = `\n          <div class="empty">\n            この分類の予想はまだありません\n          </div>\n        `;\n      }\n\n      const gradeContent =\n        document.getElementById("gradeContent");\n\n      gradeContent.innerHTML = `\n        <div class="meta history">\n          <b>${labels[grade]}・場ごとの成績</b>\n          <div>${venueHtml}</div>\n        </div>\n\n        <div class="meta history">\n          <b>${labels[grade]}・最近の予想</b>\n          ${recentHtml}\n        </div>\n      `;\n\n      resultBox\n        .querySelectorAll(".grade-tab")\n        .forEach((button) => {\n          button.classList.toggle(\n            "active",\n            button.dataset.grade === grade\n          );\n        });\n\n      gradeContent\n        .querySelectorAll(".history-row")\n        .forEach((row) => {\n          if (!row.querySelector(".history-detail")) {\n            return;\n          }\n\n          row.addEventListener("click", () => {\n            row.classList.toggle("open");\n\n            const arrow = row.querySelector(\n              ".history-arrow"\n            );\n\n            if (arrow) {\n              arrow.textContent =\n                row.classList.contains("open")\n                  ? "⌄"\n                  : "›";\n            }\n          });\n        });\n    };\n\n    const comparisonRows = [\n      ["Top1", "top1"],\n      ["Top3", "top3"],\n      ["Top5", "top5"],\n      ["Top10", "top10"],\n      ["Top20", "top20"]\n    ];\n\n    const comparisonHtml = Number(comparison.races || 0) === 0\n      ? `<div class="meta history"><b>新旧モデル比較</b><br><span class="history-muted">比較データを蓄積中です</span></div>`\n      : `<div class="meta history">\n          <b>新旧モデル比較</b><br>\n          <span class="history-muted">同じ ${Number(comparison.races || 0)} レースで比較</span>\n          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:12px;align-items:center;">\n            <b>範囲</b><b>旧モデル</b><b>新モデル v1.6</b>\n            ${comparisonRows.map(([label, key]) => `\n              <span>${label}</span>\n              <span>${pct(comparison.old ? comparison.old[key] : null)}</span>\n              <span>${pct(comparison.new ? comparison.new[key] : null)}</span>\n            `).join("")}\n          </div>\n\n          <div style="margin-top:16px;border-top:1px solid var(--line);padding-top:12px;">\n            <b>仮想回収率</b><br>\n            <span class="history-muted">新モデル：最新2回の低いオッズ・変動2倍以内・EV1.0以上・最大6点・各100円</span>\n\n            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px;align-items:center;">\n              <b>項目</b><b>旧モデル</b><b>新モデル v1.6</b>\n              <span>購入レース</span>\n              <span>${Number(comparison.old_virtual?.bet_races || 0)}</span>\n              <span>${Number(comparison.new_virtual?.bet_races || 0)}</span>\n              <span>的中レース</span>\n              <span>${Number(comparison.old_virtual?.hit_races || 0)}</span>\n              <span>${Number(comparison.new_virtual?.hit_races || 0)}</span>\n              <span>投資</span>\n              <span>${yen(comparison.old_virtual?.stake || 0)}</span>\n              <span>${yen(comparison.new_virtual?.stake || 0)}</span>\n              <span>払戻</span>\n              <span>${yen(comparison.old_virtual?.payout || 0)}</span>\n              <span>${yen(comparison.new_virtual?.payout || 0)}</span>\n              <span>収支</span>\n              <span>${yen(comparison.old_virtual?.profit || 0)}</span>\n              <span>${yen(comparison.new_virtual?.profit || 0)}</span>\n              <span>回収率</span>\n              <b>${pct(comparison.old_virtual?.roi)}</b>\n              <b>${pct(comparison.new_virtual?.roi)}</b>\n            </div>\n          </div>\n        </div>`;\n\n    const shadowHistoryHtml = shadowHistory.length === 0\n      ? `<div class="meta history"><b>新モデル直前予想履歴</b><br><span class="history-muted">結果確定データを蓄積中です</span></div>`\n      : `\n        <div class="meta history">\n          <b>新モデル直前予想履歴</b><br>\n          <span class="history-muted">EV1.0以上・最大6点</span>\n          ${shadowHistory.map(item => {\n            const title = `${item.date || \'\'} ${item.venue || \'\'} ${item.race_no ? item.race_no + \'R\' : \'\'}`.trim();\n            const timing = item.minutes_before_deadline == null\n              ? \'取得時刻不明\'\n              : `締切 ${Number(item.minutes_before_deadline).toFixed(1)}分前`;\n            const fetched = item.fetched_at\n              ? new Date(item.fetched_at).toLocaleString(\'ja-JP\')\n              : \'-\';\n            const verdict = item.hit ? \'的中\' : \'外れ\';\n            return `\n              <details style="margin-top:12px;border-top:1px solid var(--line);padding-top:10px;">\n                <summary style="cursor:pointer;font-weight:700;">${title || \'レース\'}\u3000${verdict}</summary>\n                <div class="history-muted" style="margin-top:6px;">${timing}／取得 ${fetched}</div>\n                <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px;margin-top:10px;align-items:center;">\n                  <b>買い目</b><b>確率</b><b>予想時オッズ</b><b>EV</b>\n                  ${(item.picks || []).map(pick => `\n                    <span>${pick.combo || \'-\'}</span>\n                    <span>${pct(pick.probability)}</span>\n                    <span>${pick.odds == null ? \'-\' : Number(pick.odds).toFixed(1) + \'倍\'}</span>\n                    <span>${pick.EV == null ? \'-\' : Number(pick.EV).toFixed(2)}</span>\n                  `).join(\'\') || \'<span style="grid-column:1 / -1;">対象買い目なし</span>\'}\n                </div>\n                <div style="margin-top:10px;">結果：${item.result_combo || \'-\'}\u3000払戻：${yen(item.payout || 0)}\u3000判定：<b>${verdict}</b></div>\n              </details>`;\n          }).join(\'\')}\n        </div>`;\n\n    resultBox.innerHTML = `\n      <div class="stats-grid">\n        <div class="stat">\n          <small>保存予想</small>\n          <strong>${all.saved_predictions || 0}</strong>\n        </div>\n        <div class="stat">\n          <small>結果確認済み</small>\n          <strong>${all.finished_races || 0}</strong>\n        </div>\n        <div class="stat">\n          <small>的中レース</small>\n          <strong>${all.hit_races || 0}</strong>\n        </div>\n        <div class="stat">\n          <small>的中率</small>\n          <strong>${pct(all.hit_rate)}</strong>\n        </div>\n        <div class="stat">\n          <small>収支</small>\n          <strong>${yen(all.total_profit)}</strong>\n        </div>\n        <div class="stat">\n          <small>回収率</small>\n          <strong>${pct(all.roi)}</strong>\n        </div>\n      </div>\n\n      ${comparisonHtml}\n\n      ${shadowHistoryHtml}\n\n      <div class="grade-tabs">\n        ${Object.keys(labels).map((key) => `\n          <button\n            type="button"\n            class="grade-tab"\n            data-grade="${esc(key)}"\n          >\n            ${labels[key]}\n          </button>\n        `).join("")}\n      </div>\n\n      <div id="gradeContent"></div>\n    `;\n\n    resultBox\n      .querySelectorAll(".grade-tab")\n      .forEach((button) => {\n        button.addEventListener("click", () => {\n          renderGradeSection(\n            button.dataset.grade || "all"\n          );\n        });\n      });\n\n    renderGradeSection("all");\n\n    statusBox.textContent = "累計成績";\n\n  } catch (err) {\n    statusBox.textContent = "エラー";\n    resultBox.innerHTML = `\n      <div class="empty">${esc(err.message)}</div>\n    `;\n  }\n});\n\nrefreshBtn.addEventListener("click", async () => {\n  refreshBtn.disabled = true;\n  statusBox.textContent = "結果を確認中…";\n  resultBox.innerHTML = "";\n\n  try {\n    const res = await fetch("/check-results", {\n      method: "POST"\n    });\n\n    const data = await res.json();\n\n    if (!res.ok) {\n      throw new Error(\n        data.message || data.error || "結果更新に失敗しました"\n      );\n    }\n\n    const checked = Number(\n      data.checked ?? data.updated ?? data.processed ?? 0\n    );\n\n    statusBox.textContent = "結果更新完了";\n\n    resultBox.innerHTML = `\n      <div class="empty">\n        結果確認が完了しました\n        ${checked ? `<br>${checked}件を確認・更新` : ""}\n      </div>\n    `;\n\n    // 更新後に成績を自動で再表示\n    await statsBtn.click();\n  } catch (err) {\n    statusBox.textContent = "エラー";\n    resultBox.innerHTML = `\n      <div class="empty">${esc(err.message)}</div>\n    `;\n  } finally {\n    refreshBtn.disabled = false;\n  }\n});\n\n\ndeleteBtn.addEventListener("click", async () => {\n  const ok = confirm(\n    "予想ログ・結果・収支を全部削除します。元に戻せません。削除しますか？"\n  );\n\n  if (!ok) return;\n\n  const secondCheck = confirm(\n    "本当に全削除しますか？"\n  );\n\n  if (!secondCheck) return;\n\n  statusBox.textContent = "削除中…";\n  resultBox.innerHTML = "";\n\n  try {\n    const res = await fetch("/delete-all", {\n      method: "POST"\n    });\n\n    const data = await res.json();\n\n    if (!res.ok) {\n      throw new Error(\n        data.message || data.error || "削除に失敗しました"\n      );\n    }\n\n    statusBox.textContent = "全削除完了";\n    resultBox.innerHTML = `\n      <div class="empty">\n        ${Number(data.deleted || 0)}件を削除しました\n      </div>\n    `;\n  } catch (err) {\n    statusBox.textContent = "エラー";\n    resultBox.innerHTML = `\n      <div class="empty">${esc(err.message)}</div>\n    `;\n  }\n});\n\n</script>\n</body>\n</html>\n'

@app.get('/')
def index():
    return render_template_string(HTML_PAGE)

@app.get('/health')
def health():
    try:
        load_assets()
        return jsonify({'status': 'ok', 'models_loaded': True})
    except Exception as e:
        return (jsonify({'status': 'error', 'error': str(e)}), 500)

@app.get('/predict')
def predict():
    race_text = request.args.get('race', '').strip()
    date_text = request.args.get('date')
    if not race_text:
        return (jsonify({'error': 'race is required, e.g. 蒲郡12R'}), 400)
    try:
        result = predict_internal(race_text, date_text)
        result['log_id'] = save_prediction_log(result)
        return jsonify(result)
    except Exception as e:
        return (jsonify({'error': type(e).__name__, 'message': str(e)}), 500)

@app.route('/auto-predict', methods=['GET', 'POST'])
def auto_predict():
    try:
        return jsonify(run_auto_predictions())
    except Exception as e:
        return (jsonify({'error': type(e).__name__, 'message': str(e)}), 500)

@app.route('/shadow-refresh', methods=['GET', 'POST'])
def shadow_refresh():
    try:
        return jsonify(refresh_due_shadow_predictions())
    except Exception as e:
        return (jsonify({'error': type(e).__name__, 'message': str(e)}), 500)

@app.get('/shadow-odds-check')
def shadow_odds_check():
    try:
        race_text = request.args.get('race', '').strip()
        date_text = request.args.get('date', '').strip()
        if not race_text or not date_text:
            return (jsonify({'status': 'error', 'message': 'raceとdateを指定してください', 'example': '/shadow-odds-check?race=蒲郡12R&date=2026-07-17'}), 400)
        venue_code, race_no = parse_race_input(race_text)
        target_date = date.fromisoformat(date_text)
        hd = target_date.strftime('%Y%m%d')
        doc_id = f'{target_date.isoformat()}_{venue_code:02d}_{race_no:02d}'
        db = get_firestore()
        snap = db.collection('predictions').document(doc_id).get()
        if not snap.exists:
            return (jsonify({'status': 'error', 'message': 'Firestoreに対象レースがありません', 'doc_id': doc_id}), 404)
        saved = snap.to_dict() or {}
        shadow = saved.get('shadow_model') or {}
        saved_items = shadow.get('all_combo_predictions') or []
        saved_map = {}
        for item in saved_items:
            if not isinstance(item, dict):
                continue
            combo = str(item.get('combo') or '')
            if combo:
                saved_map[combo] = item
        odds_html, _ = get_html('odds3t', hd, venue_code, race_no)
        official_df = parse_odds_table(odds_html)
        official_map = {str(row.combo): None if pd.isna(row.odds) else float(row.odds) for row in official_df.itertuples()}
        rows = []
        mismatch_count = 0
        missing_count = 0
        all_combos = sorted(set(saved_map) | set(official_map))
        for combo in all_combos:
            item = saved_map.get(combo) or {}
            saved_odds = item.get('odds')
            official_odds = official_map.get(combo)
            difference = None
            ratio = None
            mismatch = False
            if saved_odds is None or official_odds is None:
                missing_count += 1
            else:
                saved_odds = float(saved_odds)
                official_odds = float(official_odds)
                difference = round(official_odds - saved_odds, 1)
                ratio = round(official_odds / saved_odds, 3) if saved_odds else None
                mismatch = abs(difference) > 0.11
                if mismatch:
                    mismatch_count += 1
            probability = item.get('probability')
            ev = item.get('EV')
            rows.append({'combo': combo, 'probability': probability, 'saved_odds': saved_odds, 'official_current_odds': official_odds, 'difference': difference, 'ratio': ratio, 'saved_EV': ev, 'mapping_present_both': combo in saved_map and combo in official_map, 'odds_changed': mismatch})
        rows.sort(key=lambda item: (item.get('saved_EV') is not None, item.get('saved_EV') or -1), reverse=True)
        return jsonify({'status': 'ok', 'race': race_text, 'date': date_text, 'doc_id': doc_id, 'shadow_fetched_at': shadow.get('fetched_at') or saved.get('shadow_fetched_at'), 'minutes_before_deadline': shadow.get('minutes_before_deadline') or saved.get('shadow_minutes_before_deadline'), 'saved_combo_count': len(saved_map), 'official_combo_count': len(official_map), 'mapping_complete': len(saved_map) == 120 and len(official_map) == 120, 'missing_count': missing_count, 'changed_odds_count': mismatch_count, 'note': '公式現在オッズは再取得時点の値です。予想時から変動していれば差が出ます。買い目の紐付け確認はcomboと件数を見てください。', 'top_saved_ev_rows': rows[:30]})
    except Exception as exc:
        return (jsonify({'status': 'error', 'error': type(exc).__name__, 'message': str(exc)}), 500)

def count_unchecked_predictions():
    db = get_firestore()
    count = 0
    for doc in db.collection('predictions').stream():
        data = doc.to_dict() or {}
        if data.get('result_checked') is not True:
            count += 1
    return count

def run_result_backfill(max_rounds=20):
    max_rounds = max(1, min(int(max_rounds), 50))
    rounds = []
    before = count_unchecked_predictions()
    previous_remaining = before
    for round_no in range(1, max_rounds + 1):
        try:
            result = update_pending_results()
        except Exception as exc:
            rounds.append({'round': round_no, 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})
            break
        remaining = count_unchecked_predictions()
        progressed = remaining < previous_remaining
        rounds.append({'round': round_no, 'status': 'ok', 'remaining': remaining, 'progressed': progressed, 'result': result})
        if remaining == 0:
            break
        if not progressed:
            break
        previous_remaining = remaining
    after = count_unchecked_predictions()
    return {'status': 'ok', 'unchecked_before': before, 'unchecked_after': after, 'recovered': before - after, 'rounds_run': len(rounds), 'rounds': rounds, 'note': '進捗が止まった場合は、残件が未開催・結果未掲載・取得エラーの可能性があります'}

@app.route('/backfill-results', methods=['GET', 'POST'])
def backfill_results():
    try:
        max_rounds = request.args.get('rounds', '20')
        return jsonify(run_result_backfill(max_rounds=max_rounds))
    except Exception as exc:
        return (jsonify({'status': 'error', 'error': type(exc).__name__, 'message': str(exc)}), 500)

@app.route('/backfill-results-direct', methods=['GET', 'POST'])
def backfill_results_direct():
    try:
        requested = int(request.args.get('scan', '1000'))
        scan_limit = max(200, min(requested, 1000))
        before = count_unchecked_predictions()
        result = update_pending_results(limit=scan_limit)
        after = count_unchecked_predictions()
        return jsonify({'status': 'ok', 'scan_limit': scan_limit, 'unchecked_before': before, 'unchecked_after': after, 'recovered_real': before - after, 'update_result': result, 'run_again': bool(after > 0 and before > after)})
    except Exception as exc:
        return (jsonify({'status': 'error', 'error': type(exc).__name__, 'message': str(exc)}), 500)

@app.get('/stats')
def stats():
    try:
        return jsonify(build_stats())
    except Exception as e:
        return (jsonify({'error': type(e).__name__, 'message': str(e)}), 500)

@app.route('/check-results', methods=['GET', 'POST'])
def check_results():
    try:
        return jsonify(update_pending_results())
    except Exception as e:
        return (jsonify({'error': type(e).__name__, 'message': str(e)}), 500)

@app.post('/delete-all')
def delete_all():
    try:
        deleted = delete_all_predictions()
        return jsonify({'status': 'ok', 'deleted': deleted})
    except Exception as e:
        return (jsonify({'error': type(e).__name__, 'message': str(e)}), 500)

def build_roi_analysis():
    db = get_firestore()
    docs = list(db.collection('predictions').order_by('created_at', direction=firestore.Query.DESCENDING).limit(1000).stream())
    raw_rows = [(doc.to_dict() or {}) | {'id': doc.id} for doc in docs]
    latest_by_race = {}
    for row in raw_rows:
        key = (str(row.get('date')), int(row.get('venue_code') or 0), int(row.get('race_no') or 0))
        current = latest_by_race.get(key)
        if current is None:
            latest_by_race[key] = row
            continue
        row_time = row.get('created_at')
        current_time = current.get('created_at')
        if current_time is None or (row_time is not None and row_time > current_time):
            latest_by_race[key] = row
    rows = list(latest_by_race.values())

    def normalize_combo(value):
        text = str(value or '').strip()
        digits = [char for char in text if char.isdigit()]
        return '-'.join(digits[:3]) if len(digits) >= 3 else ''

    def get_result_combo(row):
        for field in ('result_combo', 'actual_combo', 'result_trifecta', 'trifecta_result'):
            combo = normalize_combo(row.get(field))
            if combo:
                return combo
        return ''
    finished_rows = []
    for row in rows:
        combo = get_result_combo(row)
        if combo:
            row = dict(row)
            row['_analysis_result_combo'] = combo
            finished_rows.append(row)

    def grade_of(row):
        grade = str(row.get('race_grade') or '一般').upper()
        if grade == 'SG':
            return 'SG'
        if grade in {'G1', 'GⅠ', 'ＧⅠ'}:
            return 'G1'
        if grade in {'G2', 'GⅡ', 'ＧⅡ'}:
            return 'G2'
        if grade in {'G3', 'GⅢ', 'ＧⅢ'}:
            return 'G3'
        return '一般'

    def venue_of(row):
        try:
            return VENUES.get(int(row.get('venue_code')), '不明')
        except (TypeError, ValueError):
            return '不明'

    def bucket(value, ranges):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return '不明'
        for low, high, label in ranges:
            if low <= value < high:
                return label
        return ranges[-1][2]
    ev_ranges = [(0.0, 1.0, 'EV<1.0'), (1.0, 1.1, '1.0-1.09'), (1.1, 1.2, '1.1-1.19'), (1.2, 1.3, '1.2-1.29'), (1.3, 1.5, '1.3-1.49'), (1.5, 2.0, '1.5-1.99'), (2.0, 999999, '2.0以上')]
    odds_ranges = [(0, 10, '10倍未満'), (10, 30, '10-29.9倍'), (30, 50, '30-49.9倍'), (50, 100, '50-99.9倍'), (100, 150, '100-149.9倍'), (150, 999999, '150倍以上')]
    probability_ranges = [(0, 0.005, '0.5%未満'), (0.005, 0.01, '0.5-0.99%'), (0.01, 0.02, '1.0-1.99%'), (0.02, 0.04, '2.0-3.99%'), (0.04, 1.0, '4.0%以上')]
    rank_ranges = [(1, 2, 'Top1'), (2, 4, 'Top2-3'), (4, 6, 'Top4-5'), (6, 11, 'Top6-10'), (11, 21, 'Top11-20'), (21, 121, 'Top21-120')]

    def collect_items(row, model):
        if model == 'old':
            items = row.get('all_combo_predictions') or []
            return [{'combo': normalize_combo(item.get('combo')), 'probability': item.get('safe_probability'), 'odds': item.get('odds'), 'ev': item.get('safe_EV'), 'rank': item.get('rank')} for item in items if isinstance(item, dict)]
        snapshots = row.get('shadow_odds_snapshots') or []
        if isinstance(snapshots, list) and snapshots:
            items = snapshots[-1].get('all_combo_predictions') or []
        else:
            shadow = row.get('shadow_model') or {}
            items = shadow.get('all_combo_predictions') or []
        return [{'combo': normalize_combo(item.get('combo')), 'probability': item.get('probability'), 'odds': item.get('odds'), 'ev': item.get('EV'), 'rank': item.get('rank')} for item in items if isinstance(item, dict)]

    def add_stat(store, key, hit, payout):
        item = store.setdefault(str(key), {'bets': 0, 'hits': 0, 'stake': 0, 'payout': 0})
        item['bets'] += 1
        item['hits'] += 1 if hit else 0
        item['stake'] += 100
        item['payout'] += int(payout) if hit else 0

    def finalize(store):
        result = []
        for label, item in store.items():
            stake = item['stake']
            payout = item['payout']
            result.append({'label': label, **item, 'profit': payout - stake, 'roi': payout / stake if stake else None, 'hit_rate': item['hits'] / item['bets'] if item['bets'] else None})
        result.sort(key=lambda x: (x['roi'] or 0, x['bets']), reverse=True)
        return result
    output = {}
    for model in ('old', 'new'):
        groups = {'ev': {}, 'odds': {}, 'rank': {}, 'probability': {}, 'grade': {}, 'venue': {}}
        total_bets = 0
        total_hits = 0
        total_payout = 0
        for row in finished_rows:
            result_combo = row['_analysis_result_combo']
            trifecta_payout = int(row.get('trifecta_payout') or row.get('result_payout') or 0)
            for item in collect_items(row, model):
                combo = item.get('combo')
                if not combo:
                    continue
                try:
                    ev = float(item.get('ev'))
                    odds = float(item.get('odds'))
                    probability = float(item.get('probability'))
                    rank = int(item.get('rank'))
                except (TypeError, ValueError):
                    continue
                hit = combo == result_combo
                payout = trifecta_payout if hit else 0
                total_bets += 1
                total_hits += 1 if hit else 0
                total_payout += payout
                add_stat(groups['ev'], bucket(ev, ev_ranges), hit, payout)
                add_stat(groups['odds'], bucket(odds, odds_ranges), hit, payout)
                add_stat(groups['probability'], bucket(probability, probability_ranges), hit, payout)
                add_stat(groups['rank'], bucket(rank, rank_ranges), hit, payout)
                add_stat(groups['grade'], grade_of(row), hit, payout)
                add_stat(groups['venue'], venue_of(row), hit, payout)
        total_stake = total_bets * 100
        output[model] = {'summary': {'races': len(finished_rows), 'bets': total_bets, 'hits': total_hits, 'stake': total_stake, 'payout': total_payout, 'profit': total_payout - total_stake, 'roi': total_payout / total_stake if total_stake else None}, 'groups': {key: finalize(value) for key, value in groups.items()}}
    return {'status': 'ok', 'loaded_documents': len(raw_rows), 'deduplicated_races': len(rows), 'finished_races': len(finished_rows), 'stake_rule': '各買い目100円の単純比較', 'analysis': output}

@app.get('/roi-analysis')
def roi_analysis():
    try:
        return jsonify(build_roi_analysis())
    except Exception as exc:
        return (jsonify({'status': 'error', 'error': type(exc).__name__, 'message': str(exc)}), 500)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8080'))
    app.run(host='0.0.0.0', port=port)
