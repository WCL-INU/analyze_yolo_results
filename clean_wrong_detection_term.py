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

# --- [핵심 제어] 시간 기준 설정 ---
# None으로 두면 마지막으로 성공한 작업 이후의 파일만 처리합니다.
# 특정 시점부터 재작업하고 싶다면 '2026-01-30 12:00:00' 형태로 입력하세요.
SINCE_DATETIME = '2026-01-30 22:10:00'  # 예: '2026-01-25 00:00:00'

# --- 필터 튜닝 ---
GRID_SIZE = 20          # 격자 크기
MAX_GAP = 20            # 잠깐 끊겨도 같은 물체로 볼 허용 간격

# [핵심 로직 설정]
CONSECUTIVE_THRESHOLD = 16  # "일정 프레임 동안 감지되면" 
LOOKAHEAD_FRAMES = 24       # "그 뒤로 이만큼을 봤는데 없으면" -> "아까 걔는 오탐지였구나" 하고 삭제

def get_reference_time():
    """기준이 되는 시간을 결정합니다."""
    # 1. 사용자가 코드에 직접 날짜를 지정한 경우
    if SINCE_DATETIME:
        return datetime.strptime(SINCE_DATETIME, '%Y-%m-%d %H:%M:%S').timestamp()
    
    # 2. 지정하지 않은 경우, Cleaned 폴더 내 가장 최근 파일의 시간을 기준점으로 삼음
    cleaned_files = list(CLEANED_DIR.glob("*.parquet"))
    if not cleaned_files:
        return 0  # 파일이 하나도 없으면 전체 처리
    
    # 가장 마지막에 생성/수정된 cleaned 파일의 시간을 반환
    return max(f.stat().st_mtime for f in cleaned_files)

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
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    REMOVED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 기준 시간 가져오기
    ref_time = get_reference_time()
    print(f"기준 시점: {datetime.fromtimestamp(ref_time).strftime('%Y-%m-%d %H:%M:%S')}")

    # RAW 폴더에서 기준 시간보다 새로운 파일만 필터링
    all_raw_files = list(INPUT_DIR.glob("*.parquet"))
    new_files = [f for f in all_raw_files if f.stat().st_mtime > ref_time]

    if not new_files:
        print("새로 추가된 데이터가 없습니다.")
        return

    print(f"새로 발견된 파일 {len(new_files)}개를 처리합니다.")
    
    con = duckdb.connect(database=':memory:')
    
    for p_file in new_files:
        print(f"Processing: {p_file.name} ...", end=" ", flush=True)
        try:
            # 기존 process_parquet_with_duckdb 함수 호출
            total, removed = process_parquet_with_duckdb(con, p_file)
            print(f"[OK] {total-removed}/{total} lines saved.")
        except Exception as e:
            print(f"[ERR] {e}")

    con.close()

if __name__ == "__main__":
    main()