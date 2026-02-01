import os
import duckdb
from pathlib import Path
from datetime import datetime
from pathlib import Path

# --- 설정 ---
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
REMOVED_DIR = BASE_DIR / "data" / "removed"

# --- 필터 튜닝 ---
GRID_SIZE = 20          # 격자 크기
MAX_GAP = 20            # 잠깐 끊겨도 같은 물체로 볼 허용 간격

# [핵심 로직 설정]
CONSECUTIVE_THRESHOLD = 16  # "일정 프레임 동안 감지되면" 
LOOKAHEAD_FRAMES = 24       # "그 뒤로 이만큼을 봤는데 없으면" -> "아까 걔는 오탐지였구나" 하고 삭제

def process_parquet_with_duckdb(con, p_file):
    filename = p_file.name
    
    # 1. 로드 & Grid 계산
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW raw_data AS 
        SELECT *,
               CAST((x + width/2) / {GRID_SIZE} AS INTEGER) AS gx,
               CAST((y + height/2) / {GRID_SIZE} AS INTEGER) AS gy
        FROM read_parquet('{str(p_file)}');
    """)
    
    # 2. 분석 쿼리 (Lookahead 로직 적용)
    analyze_sql = f"""
        CREATE OR REPLACE TEMP VIEW flagged_data AS
        WITH 
        -- (A) 연속된 프레임 그룹화 (Sessionization)
        lagged AS (
            SELECT *, frame - LAG(frame, 1, frame) OVER (PARTITION BY gx, gy ORDER BY frame) as diff
            FROM raw_data
        ),
        groups AS (
            SELECT *, CASE WHEN diff > {MAX_GAP} THEN 1 ELSE 0 END as is_new_group
            FROM lagged
        ),
        streak_ids AS (
            SELECT *, SUM(is_new_group) OVER (PARTITION BY gx, gy ORDER BY frame) as streak_id
            FROM groups
        ),
        
        -- (B) 그룹별 요약 (시작, 끝, 지속시간)
        streak_summary AS (
            SELECT gx, gy, streak_id,
                   MIN(frame) as s_start,
                   MAX(frame) as s_end,
                   MAX(frame) - MIN(frame) as duration
            FROM streak_ids
            GROUP BY gx, gy, streak_id
        ),
        
        -- (C) 미래 참조 (Lookahead)
        streak_analysis AS (
            SELECT *,
                   -- 같은 Grid 내에서 '다음 그룹'의 시작 시간을 가져옴
                   LEAD(s_start) OVER (PARTITION BY gx, gy ORDER BY s_start) as next_streak_start
            FROM streak_summary
        ),
        
        -- (D) 삭제 대상 선정 (Blacklist)
        removal_ids AS (
            SELECT streak_id
            FROM streak_analysis
            WHERE 
                -- 조건 1: 일정 시간 이상 지속되었고 (오탐지 후보군)
                duration >= {CONSECUTIVE_THRESHOLD}
                AND (
                    -- 조건 2: 다음번 나타날 때까지 공백이 Lookahead보다 길거나
                    (next_streak_start - s_end) > {LOOKAHEAD_FRAMES}
                    -- 조건 3: 아예 다시는 안 나타나거나 (영상 끝날 때까지 없음)
                    OR next_streak_start IS NULL
                )
        )
        
        -- (E) 최종 마킹
        SELECT r.*, 
               CASE WHEN rem.streak_id IS NOT NULL THEN TRUE ELSE FALSE END as is_removed
        FROM streak_ids r
        LEFT JOIN removal_ids rem ON r.streak_id = rem.streak_id;
    """
    con.execute(analyze_sql)
    
    # 통계 계산
    stats = con.execute("SELECT COUNT(*), SUM(CASE WHEN is_removed THEN 1 ELSE 0 END) FROM flagged_data").fetchone()
    total = stats[0]
    removed = stats[1] if stats[1] else 0
    
    if total == 0: return 0, 0
    
    # 3. 저장
    cleaned_path = CLEANED_DIR / filename
    con.execute(f"""
        COPY (SELECT * EXCLUDE (gx, gy, is_removed, diff, is_new_group, streak_id) FROM flagged_data WHERE is_removed = FALSE) 
        TO '{str(cleaned_path)}' (FORMAT 'parquet');
    """)
    
    removed_path = REMOVED_DIR / filename
    if removed > 0:
        con.execute(f"""
            COPY (SELECT * EXCLUDE (gx, gy, is_removed, diff, is_new_group, streak_id) FROM flagged_data WHERE is_removed = TRUE) 
            TO '{str(removed_path)}' (FORMAT 'parquet');
        """)
        
    return total, removed

def main():
    # 폴더 생성
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    REMOVED_DIR.mkdir(parents=True, exist_ok=True)
    
    parquet_files = list(INPUT_DIR.glob("*2025*.parquet"))
    print(f"Found {len(parquet_files)} files in {INPUT_DIR}")
    
    # DuckDB 인메모리 연결
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
            if removed > 0:
                print(f"  - Saved removed to: {REMOVED_DIR / p_file.name}")
            
        except Exception as e:
            print(f"  - Error: {e}")
            # 에러 상세 내용을 보고 싶다면 아래 주석 해제
            # import traceback
            # traceback.print_exc()

    con.close()

if __name__ == "__main__":
    main()