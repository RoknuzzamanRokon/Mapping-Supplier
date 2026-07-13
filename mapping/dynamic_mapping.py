import os
import math
import re
import json
from decimal import Decimal
from collections import defaultdict
from difflib import SequenceMatcher
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

PIPELINE_MAPPING = {
    "base_supplier": "agoda",
    "base_supplier_2": "ean",
    "base_supplier_3": "ratehawkhotel",
    "base_supplier_4": "didahotel",
    "base_supplier_5": "hotelbeds",
    "target_supplier": "goglobal",
}

# Set this to a single hotel_id to process only that hotel.
# Use None to process all pending hotels.
TARGET_HOTEL_ID = None
# TARGET_HOTEL_ID = "123939543"


table_2 = f"s_{PIPELINE_MAPPING['target_supplier']}_master"
table_3 = "mapping"


db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")

connection_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(
    connection_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=20,
    max_overflow=30,
    connect_args={
        "connect_timeout": 10,
        "read_timeout": 3600,
        "write_timeout": 3600,
    },
)


def get_base_supplier_sort_key(item):
    key, _ = item
    if key == "base_supplier":
        return 1

    suffix = key.replace("base_supplier_", "", 1)
    return int(suffix) if suffix.isdigit() else 999999


def get_base_suppliers():
    suppliers = [
        supplier
        for key, supplier in sorted(
            PIPELINE_MAPPING.items(),
            key=get_base_supplier_sort_key,
        )
        if key == "base_supplier" or key.startswith("base_supplier_")
    ]
    return [supplier for supplier in suppliers if supplier]


def get_supplier_table(supplier):
    return f"s_{supplier}_master"


SUPPLIER_NAME = PIPELINE_MAPPING["target_supplier"]
BASE_SUPPLIERS = get_base_suppliers()
if not BASE_SUPPLIERS:
    raise ValueError("PIPELINE_MAPPING must define at least one base_supplier key")
MATCH_RADIUS_KM = 25
EARTH_KM_PER_LAT_DEGREE = 111.0
TOP_HOTELS = 9

# Best-practice normalized weights.
# Final total max = sum(weights) = 1000
SCORE_WEIGHTS = {
    "country_code": 120,
    "geo": 300,  # combined location score
    "postal": 60,
    "name": 140,
    "local_name": 80,
    "property_type": 50,
    "state": 30,
    "city": 80,
    "address_1": 90,
    "address_2": 30,
    "star_rating": 20,
}

MAX_TOTAL_SCORE = sum(SCORE_WEIGHTS.values())  # 1000

# Thresholds tuned for normalized 0..1000 scale
AUTO_MATCH_THRESHOLD = 850
REVIEW_THRESHOLD = 800

HOTEL_STOPWORDS = {
    "hotel",
    "resort",
    "guesthouse",
    "inn",
    "lodge",
    "villa",
    "apartments",
    "motel",
    "hostel",
    "bangkok",
    "chiang",
    "mai",
    "phuket",
    "pattaya",
    "krabi",
    "samui",
    "hua",
    "hin",
    "the",
    "at",
    "in",
    "by",
    "near",
    "next",
    "grand",
    "royal",
    "sea",
    "view",
}

ADDRESS_ABBREVS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "soi": "soi",
    "mu": "mu",
    "tambon": "tambon",
    "subdistrict": "tambon",
    "fl": "floor",
    "f": "floor",
}

# Keep DB compatibility with your existing columns
PRIORITY_SCORE_FIELDS = (
    "country_code_bm",
    "lat_bm",
    "lon_bm",
    "postal_bm",
    "name_bm",
    "local_name_bm",
    "property_type_bm",
    "state_bm",
    "city_bm",
    "address_a_bm",
    "address_b_bm",
    "star_rating_bm",
)

ANALYSIS_HOTEL_FIELDS = (
    "hotel_id",
    "country_code",
    "lat",
    "lon",
    "postal_code",
    "name",
    "local_name",
    "property_type",
    "state",
    "city",
    "address_1",
    "address_2",
    "star_rating",
    "photo",
)


def normalize_hotel_id(hotel_id):
    if hotel_id is None:
        return None
    return str(hotel_id)


def normalize_text(value):
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_hotel_name(value):
    if value is None:
        return ""
    text = normalize_text(value)
    tokens = [t for t in text.split() if t not in HOTEL_STOPWORDS]
    return " ".join(tokens)


def normalize_address(value):
    if value is None:
        return ""
    text = normalize_text(value)
    for abbrev, full in ADDRESS_ABBREVS.items():
        text = re.sub(r"\b" + re.escape(abbrev) + r"\b", full, text)
    return text


def similarity_score(a, b):
    a = a or ""
    b = b or ""
    if not a and not b:
        return 0
    if a == b:
        return 100
    return round(SequenceMatcher(None, a, b).ratio() * 100)


def token_sort_similarity(a, b):
    a = a or ""
    b = b or ""
    a_sorted = " ".join(sorted(a.split()))
    b_sorted = " ".join(sorted(b.split()))
    return similarity_score(a_sorted, b_sorted)


def exact_or_similarity(a, b, threshold_full=95):
    score = token_sort_similarity(a, b)
    if score >= threshold_full:
        return 100
    return score


def name_prefix_bonus(a, b):
    """
    Returns a small raw-score bonus (0..20).
    Applied before normalization, then capped to 100 later.
    """
    a_tokens = normalize_hotel_name(a).split()[:3]
    b_tokens = normalize_hotel_name(b).split()[:3]

    if len(a_tokens) >= 2 and len(b_tokens) >= 2:
        prefix_a = " ".join(a_tokens[:2])
        prefix_b = " ".join(b_tokens[:2])
        prefix_sim = similarity_score(prefix_a, prefix_b)
        if prefix_sim >= 85:
            return min(20, round(prefix_sim * 0.2))
    return 0


def numeric_score(a, b, tolerance=0.1):
    if a is None or b is None:
        return 0
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return 0

    diff = abs(a - b)
    if diff == 0:
        return 100
    if diff <= tolerance:
        return 90
    if diff <= tolerance * 2:
        return 70
    return 0


def haversine_km(lat1, lon1, lat2, lon2):
    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except Exception:
        return None

    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def smooth_distance_score(distance_km):
    """
    Raw geo similarity score 0..100
    """
    if distance_km is None:
        return 0

    if distance_km <= 0:
        return 100
    elif distance_km <= MATCH_RADIUS_KM:
        return max(70, int(100 - (distance_km / MATCH_RADIUS_KM) * 30))
    return 0


def weighted_contribution(raw_score, weight):
    """
    Best practice:
    raw_score: 0..100
    weight: business importance
    result: 0..weight
    """
    raw_score = max(0, min(100, raw_score))
    return round((raw_score / 100.0) * weight, 2)


def fetch_all_target_supplier_rows_to_process():
    sql = text(f"""
        SELECT
            Id,
            hotel_id,
            name,
            local_name,
            property_type,
            star_rating,
            address_1,
            address_2,
            lat,
            lon,
            country_code,
            city,
            postal_code,
            state,
            status,
            photo
        FROM {table_2}
        WHERE country_code IS NOT NULL
          AND (status IS NULL OR status NOT IN ('new-mapping', 'mapped', 'review'))
          AND ittid IS NULL
          {f'AND hotel_id = :hotel_id' if TARGET_HOTEL_ID else ''}
        ORDER BY country_code ASC, Id ASC
        """)

    params = {}
    if TARGET_HOTEL_ID:
        params["hotel_id"] = TARGET_HOTEL_ID

    rows_by_country = defaultdict(list)
    with engine.begin() as conn:
        rows = conn.execute(sql, params).mappings().all()
        for row in rows:
            rows_by_country[row["country_code"]].append(row)
    return dict(rows_by_country)


def fetch_base_supplier_candidates(country_code, base_supplier):
    base_supplier_table = get_supplier_table(base_supplier)

    sql = text(f"""
        SELECT
            Id,
            hotel_id,
            name,
            local_name,
            property_type,
            star_rating,
            address_1,
            address_2,
            lat,
            lon,
            country_code,
            city,
            postal_code,
            state,
            photo
        FROM {base_supplier_table}
        WHERE country_code = :country_code
          AND ittid IS NOT NULL
    """)

    with engine.begin() as conn:
        return conn.execute(sql, {"country_code": country_code}).mappings().all()


def fetch_mappings_by_supplier(supplier):
    sql = text(f"""
        SELECT Id, ittid, supplier, hotel_id
        FROM {table_3}
        WHERE supplier = :supplier
        """)

    cache = {}
    with engine.connect().execution_options(stream_results=True) as conn:
        result = conn.execute(sql, {"supplier": supplier})
        for row in result.mappings():
            cache[normalize_hotel_id(row["hotel_id"])] = row
    return cache


def fetch_mapping_by_supplier_hotel(supplier, hotel_id):
    sql = text(f"""
        SELECT Id, ittid, supplier, hotel_id
        FROM {table_3}
        WHERE supplier = :supplier
          AND hotel_id = :hotel_id
        LIMIT 1
        """)

    with engine.begin() as conn:
        return (
            conn.execute(
                sql,
                {
                    "supplier": supplier,
                    "hotel_id": hotel_id,
                },
            )
            .mappings()
            .first()
        )


def fetch_max_ittid_sequence_by_country():
    sql = text(f"""
        SELECT
            LEFT(ittid, 2) AS country_code,
            COALESCE(MAX(CAST(SUBSTRING(ittid, 3, 8) AS UNSIGNED)), 0) AS last_seq
        FROM {table_3}
        WHERE ittid IS NOT NULL
          AND CHAR_LENGTH(ittid) >= 10
        GROUP BY LEFT(ittid, 2)
        """)

    with engine.begin() as conn:
        rows = conn.execute(sql).mappings().all()
        return {row["country_code"]: int(row["last_seq"] or 0) for row in rows}


class IttidGenerator:
    def __init__(self, starting_sequences):
        self.sequences = dict(starting_sequences)

    def next(self, country_code):
        next_seq = self.sequences.get(country_code, 0) + 1
        self.sequences[country_code] = next_seq
        return f"{country_code}{str(next_seq).zfill(8)}"


def insert_target_supplier_mapping_row(ittid, target_hotel_id):
    sql = text(f"""
        INSERT INTO {table_3} (
            ittid,
            supplier,
            hotel_id,
            country_code_bm,
            lat_bm,
            lon_bm,
            postal_bm,
            name_bm,
            local_name_bm,
            property_type_bm,
            state_bm,
            city_bm,
            address_a_bm,
            address_b_bm,
            star_rating_bm,
            total_bm
        )
        VALUES (
            :ittid,
            :supplier,
            :hotel_id,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        """)

    with engine.begin() as conn:
        result = conn.execute(
            sql,
            {
                "ittid": ittid,
                "supplier": SUPPLIER_NAME,
                "hotel_id": target_hotel_id,
            },
        )
        return result.lastrowid


def ensure_target_supplier_mapping(target_hotel_id, ittid, target_mapping_cache):
    hotel_key = normalize_hotel_id(target_hotel_id)
    existing = target_mapping_cache.get(hotel_key)
    if existing:
        return existing

    existing = fetch_mapping_by_supplier_hotel(SUPPLIER_NAME, target_hotel_id)
    if existing:
        target_mapping_cache[hotel_key] = existing
        return existing

    mapping_id = insert_target_supplier_mapping_row(ittid, target_hotel_id)
    record = {
        "Id": mapping_id,
        "ittid": ittid,
        "supplier": SUPPLIER_NAME,
        "hotel_id": target_hotel_id,
    }
    target_mapping_cache[normalize_hotel_id(target_hotel_id)] = record
    return record


def update_mapping(mapping_id, ittid, best_score, analysis_payload):
    sql = text(f"""
        UPDATE {table_3}
        SET
            ittid = :ittid,
            country_code_bm = :country_code_bm,
            lat_bm = :lat_bm,
            lon_bm = :lon_bm,
            postal_bm = :postal_bm,
            name_bm = :name_bm,
            local_name_bm = :local_name_bm,
            property_type_bm = :property_type_bm,
            state_bm = :state_bm,
            city_bm = :city_bm,
            address_a_bm = :address_a_bm,
            address_b_bm = :address_b_bm,
            star_rating_bm = :star_rating_bm,
            total_bm = :total_bm,
            analysis = :analysis
        WHERE Id = :mapping_id
        """)

    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "ittid": ittid,
                "country_code_bm": best_score["country_code_bm"],
                "lat_bm": best_score["lat_bm"],
                "lon_bm": best_score["lon_bm"],
                "postal_bm": best_score["postal_bm"],
                "name_bm": best_score["name_bm"],
                "local_name_bm": best_score["local_name_bm"],
                "property_type_bm": best_score["property_type_bm"],
                "state_bm": best_score["state_bm"],
                "city_bm": best_score["city_bm"],
                "address_a_bm": best_score["address_a_bm"],
                "address_b_bm": best_score["address_b_bm"],
                "star_rating_bm": best_score["star_rating_bm"],
                "total_bm": best_score["total_bm"],
                "analysis": analysis_payload,
                "mapping_id": mapping_id,
            },
        )


def update_target_supplier_status(target_row_id, status_value):
    sql = text(f"""
        UPDATE {table_2}
        SET status = :status
        WHERE Id = :row_id
        """)

    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "status": status_value,
                "row_id": target_row_id,
            },
        )


def get_zero_score_payload():
    return {
        "country_code_bm": 0,
        "lat_bm": 0,
        "lon_bm": 0,
        "postal_bm": 0,
        "name_bm": 0,
        "local_name_bm": 0,
        "property_type_bm": 0,
        "state_bm": 0,
        "city_bm": 0,
        "address_a_bm": 0,
        "address_b_bm": 0,
        "star_rating_bm": 0,
        "total_bm": 0,
        # "raw_scores": {},
        # "weighted_scores": {},
        "distance_km": None,
        "confidence": "none",
    }


def score_candidate(target_row, candidate_row):
    """
    Best-practice scoring:
    1. Hard gating by country + radius
    2. Raw similarity scores in 0..100
    3. Weighted contributions in 0..weight
    4. Total score in 0..1000
    """

    target_country = normalize_text(target_row["country_code"])
    candidate_country = normalize_text(candidate_row["country_code"])

    # Hard gate: must be same country
    if target_country != candidate_country:
        return None

    distance_km = haversine_km(
        target_row["lat"],
        target_row["lon"],
        candidate_row["lat"],
        candidate_row["lon"],
    )
    if distance_km is None or distance_km > MATCH_RADIUS_KM:
        return None

    # Raw scores: 0..100
    raw_country = 100
    raw_geo = smooth_distance_score(distance_km)
    raw_postal = exact_or_similarity(
        target_row["postal_code"],
        candidate_row["postal_code"],
    )

    raw_name = exact_or_similarity(
        normalize_hotel_name(target_row["name"]),
        normalize_hotel_name(candidate_row["name"]),
    )
    raw_name = min(
        100,
        raw_name + name_prefix_bonus(target_row["name"], candidate_row["name"]),
    )

    raw_local_name = exact_or_similarity(
        normalize_hotel_name(target_row["local_name"]),
        normalize_hotel_name(candidate_row["local_name"]),
    )
    raw_local_name = min(
        100,
        raw_local_name
        + name_prefix_bonus(target_row["local_name"], candidate_row["local_name"]),
    )

    raw_property_type = exact_or_similarity(
        target_row["property_type"],
        candidate_row["property_type"],
    )
    raw_state = exact_or_similarity(target_row["state"], candidate_row["state"])
    raw_city = exact_or_similarity(
        normalize_text(target_row["city"]),
        normalize_text(candidate_row["city"]),
    )
    raw_address_1 = exact_or_similarity(
        normalize_address(target_row["address_1"]),
        normalize_address(candidate_row["address_1"]),
    )
    raw_address_2 = exact_or_similarity(
        normalize_address(target_row["address_2"]),
        normalize_address(candidate_row["address_2"]),
    )
    raw_star_rating = numeric_score(
        target_row["star_rating"],
        candidate_row["star_rating"],
        tolerance=0.5,
    )

    raw_scores = {
        "country_code": raw_country,
        "geo": raw_geo,
        "postal": raw_postal,
        "name": raw_name,
        "local_name": raw_local_name,
        "property_type": raw_property_type,
        "state": raw_state,
        "city": raw_city,
        "address_1": raw_address_1,
        "address_2": raw_address_2,
        "star_rating": raw_star_rating,
    }

    # Weighted contributions: 0..weight
    weighted_scores = {
        "country_code": weighted_contribution(
            raw_scores["country_code"], SCORE_WEIGHTS["country_code"]
        ),
        "geo": weighted_contribution(raw_scores["geo"], SCORE_WEIGHTS["geo"]),
        "postal": weighted_contribution(raw_scores["postal"], SCORE_WEIGHTS["postal"]),
        "name": weighted_contribution(raw_scores["name"], SCORE_WEIGHTS["name"]),
        "local_name": weighted_contribution(
            raw_scores["local_name"], SCORE_WEIGHTS["local_name"]
        ),
        "property_type": weighted_contribution(
            raw_scores["property_type"], SCORE_WEIGHTS["property_type"]
        ),
        "state": weighted_contribution(raw_scores["state"], SCORE_WEIGHTS["state"]),
        "city": weighted_contribution(raw_scores["city"], SCORE_WEIGHTS["city"]),
        "address_1": weighted_contribution(
            raw_scores["address_1"], SCORE_WEIGHTS["address_1"]
        ),
        "address_2": weighted_contribution(
            raw_scores["address_2"], SCORE_WEIGHTS["address_2"]
        ),
        "star_rating": weighted_contribution(
            raw_scores["star_rating"], SCORE_WEIGHTS["star_rating"]
        ),
    }

    total_bm = round(sum(weighted_scores.values()), 2)

    # DB compatibility:
    # Your schema expects lat_bm and lon_bm separately.
    # We split the geo contribution equally across both.
    geo_half = round(weighted_scores["geo"] / 2.0, 2)

    db_scores = {
        "country_code_bm": weighted_scores["country_code"],
        "lat_bm": geo_half,
        "lon_bm": geo_half,
        "postal_bm": weighted_scores["postal"],
        "name_bm": weighted_scores["name"],
        "local_name_bm": weighted_scores["local_name"],
        "property_type_bm": weighted_scores["property_type"],
        "state_bm": weighted_scores["state"],
        "city_bm": weighted_scores["city"],
        "address_a_bm": weighted_scores["address_1"],
        "address_b_bm": weighted_scores["address_2"],
        "star_rating_bm": weighted_scores["star_rating"],
        "total_bm": total_bm,
    }

    if total_bm >= AUTO_MATCH_THRESHOLD:
        confidence = "high"
    elif total_bm >= REVIEW_THRESHOLD:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "candidate_row": candidate_row,
        "base_supplier_hotel_id": candidate_row["hotel_id"],
        **db_scores,
        "raw_scores": raw_scores,
        "weighted_scores": weighted_scores,
        "distance_km": distance_km,
        "confidence": confidence,
    }


def build_geo_index(candidates, radius_km):
    lat_step = max(radius_km / EARTH_KM_PER_LAT_DEGREE * 0.5, 0.001)
    index = defaultdict(list)

    for candidate in candidates:
        lat = candidate["lat"]
        lon = candidate["lon"]
        if lat is None or lon is None:
            continue

        bucket = geo_bucket(lat, lon, lat_step)
        index[bucket].append(candidate)

    return index, lat_step


def geo_bucket(lat, lon, lat_step):
    lat = float(lat)
    lon = float(lon)
    lon_step = max(lat_step * math.cos(math.radians(lat)), 0.0001)
    return (
        int(math.floor(lat / lat_step)),
        int(math.floor(lon / lon_step)),
    )


def get_candidate_pool(target_row, geo_index, lat_step):
    lat = target_row["lat"]
    lon = target_row["lon"]
    if lat is None or lon is None:
        return []

    base_lat, base_lon = geo_bucket(lat, lon, lat_step)
    candidates = []

    for lat_offset in range(-2, 3):
        for lon_offset in range(-2, 3):
            candidates.extend(
                geo_index.get((base_lat + lat_offset, base_lon + lon_offset), [])
            )

    return candidates


def candidate_priority_key(score_payload):
    return tuple(score_payload[field] for field in PRIORITY_SCORE_FIELDS)


def is_better_candidate(candidate, current_best):
    if current_best is None:
        return True

    if candidate["total_bm"] != current_best["total_bm"]:
        return candidate["total_bm"] > current_best["total_bm"]

    candidate_key = candidate_priority_key(candidate)
    current_key = candidate_priority_key(current_best)

    if candidate_key != current_key:
        return candidate_key > current_key

    return candidate["distance_km"] < current_best["distance_km"]


def sorted_candidate_scores(candidate_scores):
    return sorted(
        candidate_scores,
        key=lambda candidate: (
            candidate["total_bm"],
            candidate_priority_key(candidate),
            -candidate["distance_km"],
        ),
        reverse=True,
    )


def serialize_hotel_record(row, supplier_name):
    payload = {"supplier": supplier_name}
    for field in ANALYSIS_HOTEL_FIELDS:
        value = row.get(field)
        if isinstance(value, Decimal):
            value = float(value)
        payload[field] = value
    return payload


def get_candidate_mapping(candidate, candidate_supplier_name, candidate_mapping_cache):
    if not candidate_mapping_cache:
        return None

    if candidate_supplier_name in candidate_mapping_cache and isinstance(
        candidate_mapping_cache[candidate_supplier_name], dict
    ):
        supplier_mapping_cache = candidate_mapping_cache[candidate_supplier_name]
    else:
        supplier_mapping_cache = candidate_mapping_cache

    return supplier_mapping_cache.get(
        normalize_hotel_id(candidate["candidate_row"]["hotel_id"])
    )


def build_candidate_analysis_payload(
    candidate,
    fallback_candidate_supplier,
    candidate_mapping_cache,
):
    candidate_supplier_name = candidate.get(
        "candidate_supplier",
        fallback_candidate_supplier,
    )
    candidate_payload = serialize_hotel_record(
        candidate["candidate_row"],
        candidate_supplier_name,
    )
    candidate_mapping = get_candidate_mapping(
        candidate,
        candidate_supplier_name,
        candidate_mapping_cache,
    )
    candidate_payload["ittid"] = (
        candidate_mapping["ittid"] if candidate_mapping else None
    )
    candidate_payload["distance_km"] = (
        round(candidate["distance_km"], 4)
        if candidate["distance_km"] is not None
        else None
    )
    candidate_payload["confidence"] = candidate["confidence"]
    candidate_payload["similarity_score"] = candidate["total_bm"]
    # candidate_payload["raw_scores"] = candidate["raw_scores"]
    # candidate_payload["weighted_scores"] = candidate["weighted_scores"]
    return candidate_payload


def candidate_identity(candidate):
    return (
        candidate.get("candidate_supplier"),
        normalize_hotel_id(candidate["candidate_row"]["hotel_id"]),
    )


def select_analysis_candidates(candidate_scores):
    sorted_candidates = sorted_candidate_scores(candidate_scores)
    best_by_supplier = {}

    for candidate in sorted_candidates:
        supplier_name = candidate.get("candidate_supplier")
        if supplier_name and supplier_name not in best_by_supplier:
            best_by_supplier[supplier_name] = candidate

    selected = []
    selected_keys = set()

    for supplier_name in BASE_SUPPLIERS:
        candidate = best_by_supplier.get(supplier_name)
        if not candidate:
            continue
        selected.append(candidate)
        selected_keys.add(candidate_identity(candidate))
        if len(selected) >= TOP_HOTELS:
            return sorted_candidate_scores(selected)

    for candidate in sorted_candidates:
        key = candidate_identity(candidate)
        if key in selected_keys:
            continue
        selected.append(candidate)
        selected_keys.add(key)
        if len(selected) >= TOP_HOTELS:
            break

    return sorted_candidate_scores(selected)


def build_analysis_payload(
    source_row,
    source_supplier,
    source_ittid,
    candidate_scores,
    candidate_supplier,
    candidate_mapping_cache,
):
    top_matches = []
    top_matches_by_supplier = defaultdict(list)

    for candidate in select_analysis_candidates(candidate_scores):
        top_matches.append(
            build_candidate_analysis_payload(
                candidate,
                candidate_supplier,
                candidate_mapping_cache,
            )
        )

    for candidate in sorted_candidate_scores(candidate_scores):
        supplier_name = candidate.get("candidate_supplier", candidate_supplier)
        if len(top_matches_by_supplier[supplier_name]) >= TOP_HOTELS:
            continue
        top_matches_by_supplier[supplier_name].append(
            build_candidate_analysis_payload(
                candidate,
                candidate_supplier,
                candidate_mapping_cache,
            )
        )

    supplier_payload = serialize_hotel_record(source_row, source_supplier)
    supplier_payload["ittid"] = source_ittid

    return json.dumps(
        {
            "score_scale": {
                "max_total_score": MAX_TOTAL_SCORE,
                "auto_match_threshold": AUTO_MATCH_THRESHOLD,
                "review_threshold": REVIEW_THRESHOLD,
            },
            "supplier_data": supplier_payload,
            f"top_matches_{TOP_HOTELS}": top_matches,
            "top_matches_by_supplier": dict(top_matches_by_supplier),
        },
        ensure_ascii=False,
    )


def process_one_target_supplier(
    target_row,
    base_supplier,
    geo_index,
    lat_step,
    target_mapping_cache,
    base_supplier_mapping_cache,
    all_base_supplier_mapping_caches,
    cumulative_candidate_scores,
    ittid_generator,
):
    hotel_key = normalize_hotel_id(target_row["hotel_id"])
    if target_row["lat"] is None or target_row["lon"] is None:
        existing = target_mapping_cache.get(hotel_key)
        if existing:
            analysis_payload = build_analysis_payload(
                target_row,
                SUPPLIER_NAME,
                existing["ittid"],
                [],
                base_supplier,
                all_base_supplier_mapping_caches,
            )
            update_mapping(
                existing["Id"],
                existing["ittid"],
                get_zero_score_payload(),
                analysis_payload,
            )
        else:
            new_ittid = ittid_generator.next(target_row["country_code"])
            mapping = ensure_target_supplier_mapping(
                target_row["hotel_id"], new_ittid, target_mapping_cache
            )
            analysis_payload = build_analysis_payload(
                target_row,
                SUPPLIER_NAME,
                new_ittid,
                [],
                base_supplier,
                all_base_supplier_mapping_caches,
            )
            update_mapping(
                mapping["Id"], new_ittid, get_zero_score_payload(), analysis_payload
            )

        update_target_supplier_status(target_row["Id"], "new-mapping")
        print(f"    ⊘ SKIP: {SUPPLIER_NAME}#{target_row['hotel_id']} (no lat/lon)")
        return False

    best_score = None
    scored_candidates = []

    for candidate_row in get_candidate_pool(target_row, geo_index, lat_step):
        scored = score_candidate(target_row, candidate_row)
        if scored is None:
            continue

        scored["candidate_supplier"] = base_supplier
        scored_candidates.append(scored)

        if is_better_candidate(scored, best_score):
            best_score = scored

    cumulative_candidate_scores.extend(scored_candidates)
    overall_best_score = None
    for candidate in cumulative_candidate_scores:
        if is_better_candidate(candidate, overall_best_score):
            overall_best_score = candidate

    hotel_key = normalize_hotel_id(target_row["hotel_id"])
    if not best_score:
        existing = target_mapping_cache.get(hotel_key)
        if existing:
            ittid = existing["ittid"]
        else:
            ittid = ittid_generator.next(target_row["country_code"])
            existing = ensure_target_supplier_mapping(
                target_row["hotel_id"], ittid, target_mapping_cache
            )

        analysis_payload = build_analysis_payload(
            target_row,
            SUPPLIER_NAME,
            ittid,
            cumulative_candidate_scores,
            base_supplier,
            all_base_supplier_mapping_caches,
        )
        score_payload = (
            overall_best_score if overall_best_score else get_zero_score_payload()
        )
        update_mapping(existing["Id"], ittid, score_payload, analysis_payload)
        update_target_supplier_status(target_row["Id"], "new-mapping")
        print(
            f"    ❌ NO MATCH: {SUPPLIER_NAME}#{target_row['hotel_id']} → created ittid:{ittid}"
        )
        return False

    # Collision check
    base_supplier_existing = base_supplier_mapping_cache.get(
        normalize_hotel_id(best_score["base_supplier_hotel_id"])
    )
    if base_supplier_existing:
        existing_target = target_mapping_cache.get(
            normalize_hotel_id(base_supplier_existing["hotel_id"])
        )
        if (
            existing_target
            and normalize_hotel_id(existing_target["hotel_id"]) != hotel_key
        ):
            # Reduce confidence slightly if reused ambiguously
            if best_score["total_bm"] < 920:
                best_score["total_bm"] = round(best_score["total_bm"] * 0.95, 2)
                if (
                    best_score["confidence"] == "high"
                    and best_score["total_bm"] < AUTO_MATCH_THRESHOLD
                ):
                    best_score["confidence"] = "medium"

    overall_best_score = None
    for candidate in cumulative_candidate_scores:
        if is_better_candidate(candidate, overall_best_score):
            overall_best_score = candidate

    if overall_best_score and overall_best_score["total_bm"] >= AUTO_MATCH_THRESHOLD:
        overall_best_supplier = overall_best_score.get(
            "candidate_supplier", base_supplier
        )
        overall_supplier_mapping_cache = all_base_supplier_mapping_caches.get(
            overall_best_supplier, {}
        )
        base_supplier_map = overall_supplier_mapping_cache.get(
            normalize_hotel_id(overall_best_score["base_supplier_hotel_id"])
        )
        matched_ittid = (
            base_supplier_map["ittid"]
            if base_supplier_map
            else ittid_generator.next(target_row["country_code"])
        )

        mapping = ensure_target_supplier_mapping(
            target_row["hotel_id"], matched_ittid, target_mapping_cache
        )
        analysis_payload = build_analysis_payload(
            target_row,
            SUPPLIER_NAME,
            matched_ittid,
            cumulative_candidate_scores,
            base_supplier,
            all_base_supplier_mapping_caches,
        )
        update_mapping(
            mapping["Id"], matched_ittid, overall_best_score, analysis_payload
        )
        update_target_supplier_status(target_row["Id"], "mapped")

        print(
            f"    ✅ AUTO-MATCH: {SUPPLIER_NAME}#{target_row['hotel_id']} → {overall_best_supplier.upper()}#{overall_best_score['base_supplier_hotel_id']} | "
            f"ittid:{matched_ittid} | score:{overall_best_score['total_bm']:.1f}/{MAX_TOTAL_SCORE} | "
            f"dist:{overall_best_score['distance_km']:.2f}km | conf:{overall_best_score['confidence'].upper()}"
        )
        return True

    if overall_best_score and overall_best_score["total_bm"] >= REVIEW_THRESHOLD:
        existing = target_mapping_cache.get(hotel_key)
        if existing:
            ittid = existing["ittid"]
        else:
            ittid = ittid_generator.next(target_row["country_code"])
            existing = ensure_target_supplier_mapping(
                target_row["hotel_id"], ittid, target_mapping_cache
            )

        analysis_payload = build_analysis_payload(
            target_row,
            SUPPLIER_NAME,
            ittid,
            cumulative_candidate_scores,
            base_supplier,
            all_base_supplier_mapping_caches,
        )
        update_mapping(existing["Id"], ittid, overall_best_score, analysis_payload)
        update_target_supplier_status(target_row["Id"], "review")

        overall_best_supplier = overall_best_score.get(
            "candidate_supplier", base_supplier
        )
        print(
            f"    🟡 REVIEW: {SUPPLIER_NAME}#{target_row['hotel_id']} → {overall_best_supplier.upper()}#{overall_best_score['base_supplier_hotel_id']} | "
            f"ittid:{ittid} | score:{overall_best_score['total_bm']:.1f}/{MAX_TOTAL_SCORE} | "
            f"dist:{overall_best_score['distance_km']:.2f}km | conf:{overall_best_score['confidence'].upper()}"
        )
        return False

    existing = target_mapping_cache.get(hotel_key)
    if existing:
        ittid = existing["ittid"]
    else:
        ittid = ittid_generator.next(target_row["country_code"])
        existing = ensure_target_supplier_mapping(
            target_row["hotel_id"], ittid, target_mapping_cache
        )

    analysis_payload = build_analysis_payload(
        target_row,
        SUPPLIER_NAME,
        ittid,
        cumulative_candidate_scores,
        base_supplier,
        all_base_supplier_mapping_caches,
    )
    update_mapping(existing["Id"], ittid, overall_best_score, analysis_payload)
    update_target_supplier_status(target_row["Id"], "new-mapping")

    overall_best_supplier = overall_best_score.get("candidate_supplier", base_supplier)
    print(
        f"    ❌ WEAK MATCH: {SUPPLIER_NAME}#{target_row['hotel_id']} → {overall_best_supplier.upper()}#{overall_best_score['base_supplier_hotel_id']} | "
        f"ittid:{ittid} | score:{overall_best_score['total_bm']:.1f}/{MAX_TOTAL_SCORE} | "
        f"dist:{overall_best_score['distance_km']:.2f}km | conf:{overall_best_score['confidence'].upper()}"
    )
    return False


def best_score_from_candidates(candidate_scores):
    best_score = None
    for candidate in candidate_scores:
        if is_better_candidate(candidate, best_score):
            best_score = candidate
    return best_score


def score_target_against_base_supplier(
    target_row,
    base_supplier,
    geo_index,
    lat_step,
    target_mapping_cache,
    base_supplier_mapping_cache,
):
    best_score = None
    scored_candidates = []

    for candidate_row in get_candidate_pool(target_row, geo_index, lat_step):
        scored = score_candidate(target_row, candidate_row)
        if scored is None:
            continue

        scored["candidate_supplier"] = base_supplier
        scored_candidates.append(scored)

        if is_better_candidate(scored, best_score):
            best_score = scored

    if best_score:
        hotel_key = normalize_hotel_id(target_row["hotel_id"])
        base_supplier_existing = base_supplier_mapping_cache.get(
            normalize_hotel_id(best_score["base_supplier_hotel_id"])
        )
        if base_supplier_existing:
            existing_target = target_mapping_cache.get(
                normalize_hotel_id(base_supplier_existing["hotel_id"])
            )
            if (
                existing_target
                and normalize_hotel_id(existing_target["hotel_id"]) != hotel_key
            ):
                if best_score["total_bm"] < 920:
                    best_score["total_bm"] = round(best_score["total_bm"] * 0.95, 2)
                    if (
                        best_score["confidence"] == "high"
                        and best_score["total_bm"] < AUTO_MATCH_THRESHOLD
                    ):
                        best_score["confidence"] = "medium"

    return best_score, scored_candidates


def save_target_supplier_result(
    target_row,
    overall_best_score,
    cumulative_candidate_scores,
    target_mapping_cache,
    base_supplier_mapping_caches,
    ittid_generator,
):
    hotel_key = normalize_hotel_id(target_row["hotel_id"])

    if target_row["lat"] is None or target_row["lon"] is None:
        existing = target_mapping_cache.get(hotel_key)
        if existing:
            ittid = existing["ittid"]
        else:
            ittid = ittid_generator.next(target_row["country_code"])
            existing = ensure_target_supplier_mapping(
                target_row["hotel_id"], ittid, target_mapping_cache
            )

        analysis_payload = build_analysis_payload(
            target_row,
            SUPPLIER_NAME,
            ittid,
            [],
            BASE_SUPPLIERS[-1],
            base_supplier_mapping_caches,
        )
        update_mapping(existing["Id"], ittid, get_zero_score_payload(), analysis_payload)
        update_target_supplier_status(target_row["Id"], "new-mapping")
        print(f"    ⊘ SKIP: {SUPPLIER_NAME}#{target_row['hotel_id']} (no lat/lon)")
        return False

    if not overall_best_score:
        existing = target_mapping_cache.get(hotel_key)
        if existing:
            ittid = existing["ittid"]
        else:
            ittid = ittid_generator.next(target_row["country_code"])
            existing = ensure_target_supplier_mapping(
                target_row["hotel_id"], ittid, target_mapping_cache
            )

        analysis_payload = build_analysis_payload(
            target_row,
            SUPPLIER_NAME,
            ittid,
            cumulative_candidate_scores,
            BASE_SUPPLIERS[-1],
            base_supplier_mapping_caches,
        )
        update_mapping(existing["Id"], ittid, get_zero_score_payload(), analysis_payload)
        update_target_supplier_status(target_row["Id"], "new-mapping")
        print(
            f"    ❌ NO MATCH: {SUPPLIER_NAME}#{target_row['hotel_id']} → created ittid:{ittid}"
        )
        return False

    overall_best_supplier = overall_best_score.get("candidate_supplier", BASE_SUPPLIERS[-1])

    if overall_best_score["total_bm"] >= AUTO_MATCH_THRESHOLD:
        overall_supplier_mapping_cache = base_supplier_mapping_caches.get(
            overall_best_supplier, {}
        )
        base_supplier_map = overall_supplier_mapping_cache.get(
            normalize_hotel_id(overall_best_score["base_supplier_hotel_id"])
        )
        existing = target_mapping_cache.get(hotel_key)
        if base_supplier_map:
            ittid = base_supplier_map["ittid"]
        elif existing:
            ittid = existing["ittid"]
        else:
            ittid = ittid_generator.next(target_row["country_code"])

        mapping = ensure_target_supplier_mapping(
            target_row["hotel_id"], ittid, target_mapping_cache
        )
        analysis_payload = build_analysis_payload(
            target_row,
            SUPPLIER_NAME,
            ittid,
            cumulative_candidate_scores,
            overall_best_supplier,
            base_supplier_mapping_caches,
        )
        update_mapping(mapping["Id"], ittid, overall_best_score, analysis_payload)
        update_target_supplier_status(target_row["Id"], "mapped")

        print(
            f"    ✅ AUTO-MATCH: {SUPPLIER_NAME}#{target_row['hotel_id']} → {overall_best_supplier.upper()}#{overall_best_score['base_supplier_hotel_id']} | "
            f"ittid:{ittid} | score:{overall_best_score['total_bm']:.1f}/{MAX_TOTAL_SCORE} | "
            f"dist:{overall_best_score['distance_km']:.2f}km | conf:{overall_best_score['confidence'].upper()}"
        )
        return True

    existing = target_mapping_cache.get(hotel_key)
    if existing:
        ittid = existing["ittid"]
    else:
        ittid = ittid_generator.next(target_row["country_code"])
        existing = ensure_target_supplier_mapping(
            target_row["hotel_id"], ittid, target_mapping_cache
        )

    analysis_payload = build_analysis_payload(
        target_row,
        SUPPLIER_NAME,
        ittid,
        cumulative_candidate_scores,
        overall_best_supplier,
        base_supplier_mapping_caches,
    )

    if overall_best_score["total_bm"] >= REVIEW_THRESHOLD:
        update_mapping(existing["Id"], ittid, overall_best_score, analysis_payload)
        update_target_supplier_status(target_row["Id"], "review")
        print(
            f"    🟡 REVIEW: {SUPPLIER_NAME}#{target_row['hotel_id']} → {overall_best_supplier.upper()}#{overall_best_score['base_supplier_hotel_id']} | "
            f"ittid:{ittid} | score:{overall_best_score['total_bm']:.1f}/{MAX_TOTAL_SCORE} | "
            f"dist:{overall_best_score['distance_km']:.2f}km | conf:{overall_best_score['confidence'].upper()}"
        )
        return False

    update_mapping(existing["Id"], ittid, overall_best_score, analysis_payload)
    update_target_supplier_status(target_row["Id"], "new-mapping")
    print(
        f"    ❌ WEAK MATCH: {SUPPLIER_NAME}#{target_row['hotel_id']} → {overall_best_supplier.upper()}#{overall_best_score['base_supplier_hotel_id']} | "
        f"ittid:{ittid} | score:{overall_best_score['total_bm']:.1f}/{MAX_TOTAL_SCORE} | "
        f"dist:{overall_best_score['distance_km']:.2f}km | conf:{overall_best_score['confidence'].upper()}"
    )
    return False


def main():
    supplier_priority = " > ".join(supplier.upper() for supplier in BASE_SUPPLIERS)
    print(
        f"\n🚀 Starting hotel matching pipeline: {supplier_priority} → {SUPPLIER_NAME.upper()}"
    )
    print(
        f"⚙️  Configuration: TOP_HOTELS={TOP_HOTELS}, MATCH_RADIUS_KM={MATCH_RADIUS_KM}"
    )
    print("📂 Loading data from database...\n")

    target_rows_by_country = fetch_all_target_supplier_rows_to_process()
    target_countries = sorted(target_rows_by_country)
    if not target_countries:
        print(f"❌ No {SUPPLIER_NAME.upper()} country codes found to process")
        return

    total_rows = sum(len(rows) for rows in target_rows_by_country.values())
    print(
        f"✅ Found {len(target_countries)} countries with {total_rows} {SUPPLIER_NAME.upper()} rows "
        f"pending matching against {len(BASE_SUPPLIERS)} base suppliers: {supplier_priority}\n"
    )

    target_mapping_cache = fetch_mappings_by_supplier(SUPPLIER_NAME)
    base_supplier_mapping_caches = {
        base_supplier: fetch_mappings_by_supplier(base_supplier)
        for base_supplier in BASE_SUPPLIERS
    }
    ittid_generator = IttidGenerator(fetch_max_ittid_sequence_by_country())
    matched_target_hotel_ids = set()

    for idx, country_code in enumerate(target_countries, 1):
        print("=" * 50)
        print(f"Processing Country: {country_code.upper()}")
        print("=" * 50)

        target_rows = target_rows_by_country[country_code]
        if not target_rows:
            continue

        print(
            f"▶️  [{idx:2d}/{len(target_countries)}] Country {country_code.upper()} - "
            f"{len(target_rows)} {SUPPLIER_NAME.upper()} hotels to match"
        )

        supplier_contexts = {}
        for base_supplier in BASE_SUPPLIERS:
            base_supplier_candidates = fetch_base_supplier_candidates(
                country_code,
                base_supplier,
            )
            print(
                f"  📊 Loaded {len(base_supplier_candidates)} {base_supplier.upper()} candidates for matching"
            )

            geo_index, lat_step = build_geo_index(
                base_supplier_candidates,
                MATCH_RADIUS_KM,
            )
            print(
                f"  🗺️  Geo index: radius={MATCH_RADIUS_KM}km, lat_step={lat_step:.6f}°"
            )
            print(
                f"  📈 Scoring: max={MAX_TOTAL_SCORE}, "
                f"auto≥{AUTO_MATCH_THRESHOLD}, review≥{REVIEW_THRESHOLD}"
            )

            supplier_contexts[base_supplier] = {
                "geo_index": geo_index,
                "lat_step": lat_step,
                "mapping_cache": base_supplier_mapping_caches[base_supplier],
            }

        matched_this_country = 0
        for i, target_row in enumerate(target_rows, 1):
            if i % 50 == 0:
                percent = int((i / len(target_rows)) * 100)
                print(f"  ⏳ Progress: {i:4d}/{len(target_rows)} ({percent:3d}%)")

            cumulative_candidate_scores = []
            if target_row["lat"] is not None and target_row["lon"] is not None:
                for base_supplier in BASE_SUPPLIERS:
                    supplier_context = supplier_contexts[base_supplier]
                    _, scored_candidates = score_target_against_base_supplier(
                        target_row,
                        base_supplier,
                        supplier_context["geo_index"],
                        supplier_context["lat_step"],
                        target_mapping_cache,
                        supplier_context["mapping_cache"],
                    )
                    cumulative_candidate_scores.extend(scored_candidates)

            overall_best_score = best_score_from_candidates(cumulative_candidate_scores)
            hotel_key = normalize_hotel_id(target_row["hotel_id"])
            is_matched = save_target_supplier_result(
                target_row,
                overall_best_score,
                cumulative_candidate_scores,
                target_mapping_cache,
                base_supplier_mapping_caches,
                ittid_generator,
            )
            if is_matched:
                if hotel_key not in matched_target_hotel_ids:
                    matched_this_country += 1
                matched_target_hotel_ids.add(hotel_key)

        print(f"  ✅ {country_code} complete\n")

        remaining_target_hotels = total_rows - len(matched_target_hotel_ids)
        print("=" * 50)
        print(f"Processing Country: {country_code.upper()}")
        print(f"Matched: {matched_this_country}")
        print(f"Remaining Target Hotels: {remaining_target_hotels}")
        print("=" * 50)
        print()

    print(f"🎉 All {len(target_countries)} countries processed successfully!")


if __name__ == "__main__":
    main()
