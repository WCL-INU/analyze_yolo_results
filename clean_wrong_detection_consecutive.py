import os
import duckdb
from pathlib import Path

# --- 설정 ---
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
REMOVED_DIR = BASE_DIR / "data" / "removed"

# --- 필터 설정 ---
GRID_SIZE = 20
CONSECUTIVE_THRESHOLD = 16
STD_DEV_THRESHOLD = 10.0
MAX_GAP = 20

# [NEW] ROI(관심 영역) 설정
# 설정된 범위 '밖'에 있는 데이터는 모두 제거됩니다.
Y_FILTER_TOP_LIMIT = 1000      # Y < limit (상단) 제거
X_FILTER_LEFT_LIMIT = 1000     # X < left (좌측) 제거
X_FILTER_RIGHT_LIMIT = 1640   # X > right (우측) 제거 (영상 해상도에 맞춰 설정)

def process_parquet_with_duckdb(con, p_file):
    filename = p_file.name
    
    # 1. 원본 데이터 로드
    query_base = f"""
        CREATE OR REPLACE TEMP VIEW raw_data AS 
        SELECT *,
               CAST((x + width/2) / {GRID_SIZE} AS INTEGER) AS gx,
               CAST((y + height/2) / {GRID_SIZE} AS INTEGER) AS gy
        FROM read_parquet('{str(p_file)}');
    """
    con.execute(query_base)
    
    # 2. 분석 및 필터링 로직
    analyze_sql = f"""
        CREATE OR REPLACE TEMP VIEW flagged_data AS
        WITH lagged_data AS (
            SELECT *,
                   frame - LAG(frame, 1, frame) OVER (PARTITION BY gx, gy ORDER BY frame) as diff
            FROM raw_data
        ),
        groups AS (
            SELECT *,
                   CASE WHEN diff > {MAX_GAP} THEN 1 ELSE 0 END as is_new_group
            FROM lagged_data
        ),
        streak_ids AS (
            SELECT *,
                   SUM(is_new_group) OVER (PARTITION BY gx, gy ORDER BY frame) as streak_id
            FROM groups
        ),
        aggregated AS (
            SELECT gx, gy, streak_id,
                   MAX(frame) - MIN(frame) as duration,
                   COALESCE(STDDEV(width), 0) as w_std,
                   COALESCE(STDDEV(height), 0) as h_std
            FROM streak_ids
            GROUP BY gx, gy, streak_id
        ),
        blacklisted_grids AS (
            SELECT DISTINCT gx, gy
            FROM aggregated
            WHERE duration > {CONSECUTIVE_THRESHOLD}
              AND w_std < {STD_DEV_THRESHOLD}
              AND h_std < {STD_DEV_THRESHOLD}
        )
        SELECT r.*, 
               CASE 
                   -- [수정됨] ROI(좌표) 필터링: 설정 범위 밖이면 제거
                   WHEN r.y < {Y_FILTER_TOP_LIMIT} THEN TRUE    -- 상단 제거
                   WHEN r.x < {X_FILTER_LEFT_LIMIT} THEN TRUE   -- 좌측 제거
                   WHEN r.x > {X_FILTER_RIGHT_LIMIT} THEN TRUE  -- 우측 제거
                   
                   -- 기존 정적 물체(Box Filtering) 제거
                   WHEN b.gx IS NOT NULL THEN TRUE 
                   ELSE FALSE 
               END as is_removed
        FROM raw_data r
        LEFT JOIN blacklisted_grids b ON r.gx = b.gx AND r.gy = b.gy;
    """
    con.execute(analyze_sql)
    
    # 결과 통계
    stats = con.execute("SELECT COUNT(*), SUM(CASE WHEN is_removed THEN 1 ELSE 0 END) FROM flagged_data").fetchone()
    total_count = stats[0]
    removed_count = stats[1] if stats[1] else 0
    
    # 3. 저장
    cleaned_path = CLEANED_DIR / filename
    con.execute(f"""
        COPY (SELECT * EXCLUDE (gx, gy, is_removed) FROM flagged_data WHERE is_removed = FALSE) 
        TO '{str(cleaned_path)}' (FORMAT 'parquet');
    """)
    
    removed_path = REMOVED_DIR / filename
    if removed_count > 0:
        con.execute(f"""
            COPY (SELECT * EXCLUDE (gx, gy, is_removed) FROM flagged_data WHERE is_removed = TRUE) 
            TO '{str(removed_path)}' (FORMAT 'parquet');
        """)
    
    return total_count, removed_count

def main():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    REMOVED_DIR.mkdir(parents=True, exist_ok=True)
    
    parquet_files = list(INPUT_DIR.glob("*-15_2026*.parquet"))
    print(f"Found {len(parquet_files)} files in {INPUT_DIR}")
    print(f"ROI Filter: Y>{Y_FILTER_TOP_LIMIT}, {X_FILTER_LEFT_LIMIT}<X<{X_FILTER_RIGHT_LIMIT}")
    
    con = duckdb.connect(database=':memory:')
    
    for p_file in parquet_files:
        print(f"\nProcessing: {p_file.name} ...")
        try:
            total, removed = process_parquet_with_duckdb(con, p_file)
            
            if total == 0:
                print("  - Empty file.")
                continue

            percent = (removed / total * 100) if total > 0 else 0
            print(f"  - Total: {total}, Removed: {removed} ({percent:.1f}%)")
            
        except Exception as e:
            print(f"  - Error: {e}")

    con.close()

if __name__ == "__main__":
    main()