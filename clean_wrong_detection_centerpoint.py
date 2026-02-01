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
Y_FILTER_TOP_LIMIT = 0      # Y < limit (상단) 제거
X_FILTER_LEFT_LIMIT = 0     # X < left (좌측) 제거
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
            -- 이전과 동일: 같은 격자 내에서 시간 차이 계산
            SELECT *,
                   frame - LAG(frame, 1, frame) OVER (PARTITION BY gx, gy ORDER BY frame) as diff
            FROM raw_data
        ),
        groups AS (
            -- 이전과 동일: 시간 차이가 크면 그룹 분리
            SELECT *,
                   CASE WHEN diff > {MAX_GAP} THEN 1 ELSE 0 END as is_new_group
            FROM lagged_data
        ),
        streak_ids AS (
            -- 이전과 동일: 고유 ID 생성 (streak_id)
            SELECT *,
                   SUM(is_new_group) OVER (PARTITION BY gx, gy ORDER BY frame) as streak_id
            FROM groups
        ),
        aggregated AS (
            SELECT streak_id,  -- 격자(gx, gy) 대신 streak_id가 핵심 키가 됨
                   MAX(frame) - MIN(frame) as duration,
                   -- [핵심] 중심점 좌표의 변동성 계산
                   COALESCE(STDDEV(x), 0) as x_std,
                   COALESCE(STDDEV(y), 0) as y_std,
                   -- 크기 변동성도 보조 지표로 사용
                   COALESCE(STDDEV(width), 0) as w_std,
                   COALESCE(STDDEV(height), 0) as h_std
            FROM streak_ids
            GROUP BY streak_id
        ),
        bad_streaks AS (
            SELECT streak_id
            FROM aggregated
            WHERE duration > {CONSECUTIVE_THRESHOLD}
              -- [조건 변경] 좌표가 너무 고정되어 있으면(돌) 제거
              -- 예: X, Y 표준편차가 3.0 미만이면 거의 안 움직인 것
              AND x_std < 3.0 
              AND y_std < 3.0
              -- (선택 사항) 크기 변동성 조건도 유지 가능
              AND w_std < {STD_DEV_THRESHOLD}
        )
        SELECT r.*, 
               CASE 
                   WHEN r.y < {Y_FILTER_TOP_LIMIT} THEN TRUE
                   WHEN r.x < {X_FILTER_LEFT_LIMIT} THEN TRUE
                   WHEN r.x > {X_FILTER_RIGHT_LIMIT} THEN TRUE
                   
                   -- [변경됨] 격자가 아니라, '해당 움직임(streak_id)'이 불량인지 확인
                   WHEN b.streak_id IS NOT NULL THEN TRUE 
                   ELSE FALSE 
               END as is_removed
        FROM streak_ids r  -- raw_data 대신 streak_ids를 베이스로 사용
        LEFT JOIN bad_streaks b ON r.streak_id = b.streak_id;
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
    
    parquet_files = list(INPUT_DIR.glob("*-3_20260107*.parquet"))
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