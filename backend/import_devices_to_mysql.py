"""
Import GSMArena CSV files into MySQL devices table.

Usage:
python import_devices_to_mysql.py --csv-folder ..\DatasetPhones\GSMArenaDataset --db-url mysql+pymysql://user:pass@host:3306/device_catalog
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import create_engine, text


TARGET_COLUMNS: List[str] = [
    "brand",
    "model_name",
    "model_image",
    "technology",
    "bands_2g",
    "bands_3g",
    "bands_4g",
    "bands_5g",
    "speed",
    "gprs",
    "edge",
    "announced",
    "status",
    "dimensions",
    "weight",
    "build",
    "sim",
    "keyboard",
    "display_type",
    "display_size",
    "display_resolution",
    "display_protection",
    "battery_capacity",
    "os",
    "chipset",
    "cpu",
    "gpu",
    "card_slot",
    "internal_storage",
    "phonebook",
    "call_records",
    "unknown_1",
    "triple_camera",
    "main_camera_features",
    "main_camera_video",
    "selfie_camera_single",
    "quad_camera",
    "dual_camera",
    "selfie_features",
    "selfie_single_1",
    "selfie_video",
    "unknown_1_1",
    "loudspeaker",
    "jack_35mm",
    "loudspeaker_1",
    "alert_types",
    "wlan",
    "bluetooth",
    "positioning",
    "nfc",
    "radio",
    "usb",
    "infrared_port",
    "sensors",
    "messaging",
    "browser",
    "clock",
    "alarm",
    "games",
    "languages",
    "java",
    "colors",
    "models",
    "price",
    "battery",
    "charging",
    "standby",
    "talk_time",
    "music_play",
    "battery_old",
    "standby_1",
    "talk_time_1",
    "sar",
    "sar_eu",
    "energy",
    "free_fall",
    "repairability",
    "variants",
    "antutu",
    "geekbench",
    "gfxbench",
    "display_contrast",
    "loudspeaker_lufs",
    "performance",
    "display",
    "camera",
    "loudspeaker_2",
    "audio_quality",
    "battery_life",
    "extra_col",
    "extra_col2",
    "extra_col3",
    "extra_col4",
    "extra_col5",
    "extra_col6",
    "extra_col7",
    "extra_col8",
    "extra_col9",
    "extra_col10",
    "extra_col11",
    "extra_col12",
    "extra_col13",
]


EXTRA_COLUMNS: List[str] = [
    "extra_col",
    "extra_col2",
    "extra_col3",
    "extra_col4",
    "extra_col5",
    "extra_col6",
    "extra_col7",
    "extra_col8",
    "extra_col9",
    "extra_col10",
    "extra_col11",
    "extra_col12",
    "extra_col13",
]


ALIAS_MAP: Dict[str, str] = {
    "brand": "brand",
    "modelname": "model_name",
    "model_name": "model_name",
    "modelimage": "model_image",
    "model_image": "model_image",
    "2gbands": "bands_2g",
    "3gbands": "bands_3g",
    "4gbands": "bands_4g",
    "5gbands": "bands_5g",
    "displaytype": "display_type",
    "displaysize": "display_size",
    "displayresolution": "display_resolution",
    "displayprotection": "display_protection",
    "batterycapacity": "battery_capacity",
    "cardslot": "card_slot",
    "internalstorage": "internal_storage",
    "maincamerafeatures": "main_camera_features",
    "maincameravideo": "main_camera_video",
    "selfiecamerasingle": "selfie_camera_single",
    "selfiecamera_single": "selfie_camera_single",
    "selfiefeatures": "selfie_features",
    "selfiesingle1": "selfie_single_1",
    "selfievideo": "selfie_video",
    "jack35mm": "jack_35mm",
    "infraredport": "infrared_port",
    "batteryold": "battery_old",
    "talktime": "talk_time",
    "musicplay": "music_play",
    "standby1": "standby_1",
    "talktime1": "talk_time_1",
    "sareu": "sar_eu",
    "freefall": "free_fall",
    "displaycontrast": "display_contrast",
    "loudspeakerlufs": "loudspeaker_lufs",
    "audioquality": "audio_quality",
    "batterylife": "battery_life",
}


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").strip().lower())


def to_text(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "n/a", "-"}:
        return None
    return s


def build_row_mapper(headers: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Return mapping from db column -> csv header plus unmapped headers list."""
    normalized_to_header: Dict[str, str] = {}

    for header in headers:
        key = normalize(header)
        if key and key not in normalized_to_header:
            normalized_to_header[key] = header

    mapped: Dict[str, str] = {}

    for target in TARGET_COLUMNS:
        if target in EXTRA_COLUMNS:
            continue

        n_target = normalize(target)

        if n_target in normalized_to_header:
            mapped[target] = normalized_to_header[n_target]
            continue

        if n_target in ALIAS_MAP:
            alias_target = ALIAS_MAP[n_target]
            n_alias_target = normalize(alias_target)
            if n_alias_target in normalized_to_header:
                mapped[target] = normalized_to_header[n_alias_target]
                continue

        for n_header, original in normalized_to_header.items():
            mapped_target = ALIAS_MAP.get(n_header)
            if mapped_target == target:
                mapped[target] = original
                break

    mapped_headers = set(mapped.values())
    unmapped_headers = [h for h in headers if h not in mapped_headers]
    return mapped, unmapped_headers


def create_table_if_missing(engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS devices (
        id INT AUTO_INCREMENT PRIMARY KEY,
        brand LONGTEXT,
        model_name LONGTEXT,
        model_image LONGTEXT,
        technology LONGTEXT,
        bands_2g LONGTEXT,
        bands_3g LONGTEXT,
        bands_4g LONGTEXT,
        bands_5g LONGTEXT,
        speed LONGTEXT,
        gprs LONGTEXT,
        edge LONGTEXT,
        announced LONGTEXT,
        status LONGTEXT,
        dimensions LONGTEXT,
        weight LONGTEXT,
        build LONGTEXT,
        sim LONGTEXT,
        keyboard LONGTEXT,
        display_type LONGTEXT,
        display_size LONGTEXT,
        display_resolution LONGTEXT,
        display_protection LONGTEXT,
        battery_capacity LONGTEXT,
        os LONGTEXT,
        chipset LONGTEXT,
        cpu LONGTEXT,
        gpu LONGTEXT,
        card_slot LONGTEXT,
        internal_storage LONGTEXT,
        phonebook LONGTEXT,
        call_records LONGTEXT,
        unknown_1 LONGTEXT,
        triple_camera LONGTEXT,
        main_camera_features LONGTEXT,
        main_camera_video LONGTEXT,
        selfie_camera_single LONGTEXT,
        quad_camera LONGTEXT,
        dual_camera LONGTEXT,
        selfie_features LONGTEXT,
        selfie_single_1 LONGTEXT,
        selfie_video LONGTEXT,
        unknown_1_1 LONGTEXT,
        loudspeaker LONGTEXT,
        jack_35mm LONGTEXT,
        loudspeaker_1 LONGTEXT,
        alert_types LONGTEXT,
        wlan LONGTEXT,
        bluetooth LONGTEXT,
        positioning LONGTEXT,
        nfc LONGTEXT,
        radio LONGTEXT,
        usb LONGTEXT,
        infrared_port LONGTEXT,
        sensors LONGTEXT,
        messaging LONGTEXT,
        browser LONGTEXT,
        clock LONGTEXT,
        alarm LONGTEXT,
        games LONGTEXT,
        languages LONGTEXT,
        java LONGTEXT,
        colors LONGTEXT,
        models LONGTEXT,
        price LONGTEXT,
        battery LONGTEXT,
        charging LONGTEXT,
        standby LONGTEXT,
        talk_time LONGTEXT,
        music_play LONGTEXT,
        battery_old LONGTEXT,
        standby_1 LONGTEXT,
        talk_time_1 LONGTEXT,
        sar LONGTEXT,
        sar_eu LONGTEXT,
        energy LONGTEXT,
        free_fall LONGTEXT,
        repairability LONGTEXT,
        variants LONGTEXT,
        antutu LONGTEXT,
        geekbench LONGTEXT,
        gfxbench LONGTEXT,
        display_contrast LONGTEXT,
        loudspeaker_lufs LONGTEXT,
        performance LONGTEXT,
        display LONGTEXT,
        camera LONGTEXT,
        loudspeaker_2 LONGTEXT,
        audio_quality LONGTEXT,
        battery_life LONGTEXT,
        extra_col LONGTEXT,
        extra_col2 LONGTEXT,
        extra_col3 LONGTEXT,
        extra_col4 LONGTEXT,
        extra_col5 LONGTEXT,
        extra_col6 LONGTEXT,
        extra_col7 LONGTEXT,
        extra_col8 LONGTEXT,
        extra_col9 LONGTEXT,
        extra_col10 LONGTEXT,
        extra_col11 LONGTEXT,
        extra_col12 LONGTEXT,
        extra_col13 LONGTEXT
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def build_insert_sql() -> str:
    cols = ", ".join(TARGET_COLUMNS)
    vals = ", ".join(f":{c}" for c in TARGET_COLUMNS)
    return f"INSERT INTO devices ({cols}) VALUES ({vals})"


def import_csv_file(engine, csv_path: Path, batch_size: int = 500) -> int:
    inserted_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        mapped, unmapped_headers = build_row_mapper(headers)

        rows_batch: List[Dict[str, str | None]] = []
        sql = text(build_insert_sql())

        for row in reader:
            record: Dict[str, str | None] = {c: None for c in TARGET_COLUMNS}

            for target_col, source_header in mapped.items():
                record[target_col] = to_text(row.get(source_header))

            if not record.get("brand"):
                record["brand"] = csv_path.stem

            extras: List[str] = []
            for header in unmapped_headers:
                val = to_text(row.get(header))
                if val is not None:
                    extras.append(val)

            for idx, extra_col in enumerate(EXTRA_COLUMNS):
                record[extra_col] = extras[idx] if idx < len(extras) else None

            if not record.get("model_name"):
                continue

            rows_batch.append(record)

            if len(rows_batch) >= batch_size:
                with engine.begin() as conn:
                    conn.execute(sql, rows_batch)
                inserted_count += len(rows_batch)
                rows_batch.clear()

        if rows_batch:
            with engine.begin() as conn:
                conn.execute(sql, rows_batch)
            inserted_count += len(rows_batch)

    return inserted_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Import GSMArena CSV files into MySQL devices table")
    parser.add_argument("--csv-folder", required=True, help="Folder containing GSMArena CSV files")
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL", "mysql+pymysql://root:123@localhost:3306/device_catalog"),
        help="MySQL SQLAlchemy URL",
    )
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for inserts")
    parser.add_argument(
        "--create-table",
        action="store_true",
        help="Create devices table if it does not already exist",
    )

    args = parser.parse_args()

    csv_folder = Path(args.csv_folder)
    if not csv_folder.exists() or not csv_folder.is_dir():
        raise FileNotFoundError(f"CSV folder not found: {csv_folder}")

    csv_files = sorted(csv_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_folder}")

    engine = create_engine(args.db_url, pool_pre_ping=True)

    if args.create_table:
        create_table_if_missing(engine)

    total = 0
    print(f"Found {len(csv_files)} CSV files")

    for file in csv_files:
        count = import_csv_file(engine, file, batch_size=args.batch_size)
        total += count
        print(f"Imported {count} rows from {file.name}")

    print(f"Done. Total inserted rows: {total}")


if __name__ == "__main__":
    main()
