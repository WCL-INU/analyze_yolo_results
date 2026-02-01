import os
import duckdb
from pathlib import Path

# --- 설정 ---
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data" / "raw"        # 원본 파일 폴더
CLEANED_DIR = BASE_DIR / "data" / "cleaned"  # 정상 데이터 저장 폴더
REMOVED_DIR = BASE_DIR / "data" / "removed"  # 삭제된 데이터 저장 폴더

# --- 필터 설정 ---
GRID_SIZE = 20
CONSECUTIVE_THRESHOLD = 16  # 연속성 임계값 (프레임 수)
STD_DEV_THRESHOLD = 10.0    # 변동성 임계값 (너비/높이 표준편차)
MAX_GAP = 20                # 허용 Gap

# [NEW] ROI 설정: 이 값보다 Y좌표가 작으면(영상 상단) 무조건 제거
Y_FILTER_LIMIT = 1000        # 예: Y좌표 0~300 사이의 박스는 무시 (배경/하늘 등)

def process_parquet_with_duckdb(con, p_file):
    filename = p_file.name
    
    # 1. 원본 데이터 로드 및 Grid Key 계산
    query_base = f"""
        CREATE OR REPLACE TEMP VIEW raw_data AS 
        SELECT *,
               CAST((x + width/2) / {GRID_SIZE} AS INTEGER) AS gx,
               CAST((y + height/2) / {GRID_SIZE} AS INTEGER) AS gy
        FROM read_parquet('{str(p_file)}');
    """
    con.execute(query_base)
    
    # 2. 복잡한 필터링 로직을 SQL로 구현
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
                   -- [수정됨] 1. Y좌표가 기준치 미만(상단부)이면 무조건 제거
                   WHEN r.y < {Y_FILTER_LIMIT} THEN TRUE
                   -- 2. 정적 물체(Box Filtering)로 판별되면 제거
                   WHEN b.gx IS NOT NULL THEN TRUE 
                   ELSE FALSE 
               END as is_removed
        FROM raw_data r
        LEFT JOIN blacklisted_grids b ON r.gx = b.gx AND r.gy = b.gy;
    """
    con.execute(analyze_sql)
    
    # 결과 통계 확인
    stats = con.execute("SELECT COUNT(*), SUM(CASE WHEN is_removed THEN 1 ELSE 0 END) FROM flagged_data").fetchone()
    total_count = stats[0]
    removed_count = stats[1] if stats[1] else 0
    
    # 3. 데이터 분리 저장
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
    
    parquet_files = list(INPUT_DIR.glob("*-2_2026*.parquet"))
    print(f"Found {len(parquet_files)} files in {INPUT_DIR}")
    print(f"Filtering Logic: Box Filter + Y < {Y_FILTER_LIMIT} (Top Area Removal)")
    
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
            print(f"  - Saved cleaned to: {CLEANED_DIR / p_file.name}")
            
        except Exception as e:
            print(f"  - Error: {e}")

    con.close()

if __name__ == "__main__":
    main()