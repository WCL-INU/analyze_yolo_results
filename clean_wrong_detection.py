import os
import duckdb
from pathlib import Path
from datetime import datetime

# --- 설정 ---
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data" / "raw"        # 원본 파일 폴더
CLEANED_DIR = BASE_DIR / "data" / "cleaned"  # 정상 데이터 저장 폴더
REMOVED_DIR = BASE_DIR / "data" / "removed"  # 삭제된 데이터 저장 폴더

# --- [핵심 제어] 시간 기준 설정 ---
# None: 'cleaned' 폴더의 최신 파일 이후만 처리 (증분 업데이트)
# '2026-01-30 00:00:00': 입력한 시점 이후의 원본 데이터를 모두 다시 처리
SINCE_DATETIME = '2026-01-30 22:10:00'

# --- 필터 설정 ---
GRID_SIZE = 20
CONSECUTIVE_THRESHOLD = 16  # 연속성 임계값
STD_DEV_THRESHOLD = 10.0    # 변동성 임계값
MAX_GAP = 20                # 허용 Gap

def get_reference_time():
    """작업 시작의 기준이 되는 타임스탬프를 가져옵니다."""
    # 1. 수동 설정이 있는 경우
    if SINCE_DATETIME:
        return datetime.strptime(SINCE_DATETIME, '%Y-%m-%d %H:%M:%S').timestamp()
    
    # 2. 수동 설정이 없으면 Cleaned 폴더의 최신 파일 시간 기준
    cleaned_files = list(CLEANED_DIR.glob("*.parquet"))
    if not cleaned_files:
        return 0
    return max(f.stat().st_mtime for f in cleaned_files)

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
    
    # 2. 필터링 로직 SQL
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
               CASE WHEN b.gx IS NOT NULL THEN TRUE ELSE FALSE END as is_removed
        FROM raw_data r
        LEFT JOIN blacklisted_grids b ON r.gx = b.gx AND r.gy = b.gy;
    """
    con.execute(analyze_sql)
    
    stats = con.execute("SELECT COUNT(*), SUM(CASE WHEN is_removed THEN 1 ELSE 0 END) FROM flagged_data").fetchone()
    total_count = stats[0]
    removed_count = stats[1] if stats[1] else 0
    
    # 3. 데이터 분리 저장
    if total_count > 0:
        cleaned_path = CLEANED_DIR / filename
        con.execute(f"""
            COPY (SELECT * EXCLUDE (gx, gy, is_removed) FROM flagged_data WHERE is_removed = FALSE) 
            TO '{str(cleaned_path)}' (FORMAT 'parquet');
        """)
        
        if removed_count > 0:
            removed_path = REMOVED_DIR / filename
            con.execute(f"""
                COPY (SELECT * EXCLUDE (gx, gy, is_removed) FROM flagged_data WHERE is_removed = TRUE) 
                TO '{str(removed_path)}' (FORMAT 'parquet');
            """)
    
    return total_count, removed_count

def main():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    REMOVED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 기준 시점 계산
    ref_time = get_reference_time()
    ref_dt = datetime.fromtimestamp(ref_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"--- 작업 기준 시점: {ref_dt} ---")

    # 1. 파일 목록 가져오기 (기준 시점보다 최신인 파일)
    all_files = list(INPUT_DIR.glob("*.parquet"))
    new_files = [f for f in all_files if f.stat().st_mtime > ref_time]
    
    # 시간 순으로 정렬 (선택 사항이지만 권장)
    new_files.sort(key=lambda x: x.stat().st_mtime)

    if not new_files:
        print("새로 처리할 파일이 없습니다.")
        return

    print(f"총 {len(new_files)}개의 새로운 파일을 발견했습니다.")
    
    con = duckdb.connect(database=':memory:')
    
    for p_file in new_files:
        mtime_dt = datetime.fromtimestamp(p_file.stat().st_mtime).strftime('%y-%m-%d %H:%M')
        print(f"\n[{mtime_dt}] Processing: {p_file.name} ...")
        try:
            total, removed = process_parquet_with_duckdb(con, p_file)
            if total == 0:
                print("  - Empty file.")
                continue

            percent = (removed / total * 100) if total > 0 else 0
            print(f"  - Results: Total {total} / Removed {removed} ({percent:.1f}%)")
        except Exception as e:
            print(f"  - Error: {e}")

    con.close()
    print("\n--- 모든 작업 완료 ---")

if __name__ == "__main__":
    main()