import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

table_1 = "s_agoda_master"
table_2 = "mapping"

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")

connection_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

engine = create_engine(
    connection_url, pool_pre_ping=True, pool_recycle=3600, pool_size=20, max_overflow=30
)

supplier_name = "agoda"


def get_country_codes(conn):
    country_code_sql = text(f"""
        SELECT DISTINCT country_code
        FROM {table_1}
        WHERE ittid IS NULL
          AND country_code IS NOT NULL
          AND country_code <> ''
        ORDER BY country_code
    """)
    return [row[0] for row in conn.execute(country_code_sql).fetchall()]


def transfer_data():
    with engine.begin() as conn:
        country_codes = get_country_codes(conn)

        for target_country_code in country_codes:
            print(f"\nProcessing country: {target_country_code}")

            # ==============================
            # STEP 1: GET LAST SEQUENCE
            # ==============================
            last_seq_sql = text(f"""
                SELECT COALESCE(
                    MAX(CAST(SUBSTRING(ittid, 3, 8) AS UNSIGNED)),
                    0
                )
                FROM {table_2}
                WHERE ittid LIKE :prefix
            """)

            last_seq = conn.execute(
                last_seq_sql, {"prefix": f"{target_country_code}%"}
            ).scalar()

            print(f"Last sequence for {target_country_code}: {last_seq}")

            # ==============================
            # STEP 2: INSERT INTO MAPPING
            # ==============================
            insert_sql = text(f"""
                INSERT INTO {table_2} (
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
                    total_bm,
                    verified_by
                )
                SELECT
                    CONCAT(:country_code,
                        LPAD(
                            (:last_seq + ROW_NUMBER() OVER (ORDER BY t1.id)),
                            8,
                            '0'
                        )
                    ) AS ittid,
                    :supplier AS supplier,
                    t1.hotel_id,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    'base-mapped' AS verified_by
                FROM {table_1} t1
                LEFT JOIN {table_2} m
                    ON m.hotel_id = t1.hotel_id
                    AND m.supplier = :supplier
                WHERE t1.country_code = :country_code
                  AND m.hotel_id IS NULL
            """)

            insert_result = conn.execute(
                insert_sql,
                {
                    "country_code": target_country_code,
                    "last_seq": int(last_seq),
                    "supplier": supplier_name,
                },
            )

            print(f"Inserted {insert_result.rowcount} new rows into {table_2}")

            # ==============================
            # STEP 3: UPDATE MASTER TABLE
            # ==============================
            update_sql = text(f"""
                UPDATE {table_1} t1
                JOIN {table_2} m
                    ON t1.hotel_id = m.hotel_id
                SET 
                    t1.ittid = m.ittid,
                    t1.status = 'base-mapped'
                WHERE m.supplier = :supplier
                AND t1.country_code = :country_code
                AND (
                        t1.ittid IS NULL
                        OR t1.ittid = ''
                        OR t1.ittid <> m.ittid
                    )
            """)

            update_result = conn.execute(
                update_sql,
                {"supplier": supplier_name, "country_code": target_country_code},
            )

            print(f"Updated {update_result.rowcount} rows in {table_1}")


if __name__ == "__main__":
    transfer_data()
