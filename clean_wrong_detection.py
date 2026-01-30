import os
import glob
import statistics
import pandas as pd
import numpy as np
from pathlib import Path

# --- 설정 ---
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data" / "raw"       # 원본 파일 폴더
OUTPUT_DIR = BASE_DIR / "data" / "cleaned"  # 결과 저장 폴더

# --- 필터 설정 (사용자 요청 로직) ---
GRID_SIZE = 20
CONSECUTIVE_THRESHOLD = 16  # 연속성 임계값
STD_DEV_THRESHOLD = 10.0    # 변동성 임계값
MAX_GAP = 20                # 허용 Gap

def get_grid_key(x, y, w, h):
    cx = x + w / 2
    cy = y + h / 2
    return (int(cx // GRID_SIZE), int(cy // GRID_SIZE))

def get_static_blacklist(df):
    """
    DataFrame을 분석하여 정적 물체(오탐지)에 해당하는 Grid Key 집합(Blacklist)을 반환
    """
    print(f"  - Analyzing {len(df)} rows for static objects...")
    
    # 프레임별로 데이터 그룹화 (속도 향상을 위해)
    # { frame_num: [ {row_dict}, ... ] }
    frames_data = {k: v.to_dict('records') for k, v in df.groupby('frame')}
    sorted_frames = sorted(frames_data.keys())

    # 격자별 상태 추적
    # key: (gx, gy), value: { last_frame, current_streak, max_streak, w_list, h_list }
    grid_history = {}

    for frame in sorted_frames:
        boxes = frames_data[frame]
        
        for box in boxes:
            k = get_grid_key(box['x'], box['y'], box['width'], box['height'])
            
            if k not in grid_history:
                grid_history[k] = {
                    'last_frame': frame, 
                    'current_streak': 0, 
                    'max_streak': 0,
                    'w_list': [], 'h_list': []
                }
            
            history = grid_history[k]
            frame_diff = frame - history['last_frame']
            
            # --- Gap 허용 연속성 체크 로직 ---
            if frame_diff <= MAX_GAP:
                if frame_diff > 0:
                    history['current_streak'] += frame_diff
            else:
                # 끊김 -> 리셋
                history['current_streak'] = 0
            
            # 최대 연속 기록 갱신
            if history['current_streak'] > history['max_streak']:
                history['max_streak'] = history['current_streak']
            
            # 상태 업데이트
            history['last_frame'] = frame
            history['w_list'].append(box['width'])
            history['h_list'].append(box['height'])

    # 블랙리스트 선정
    blacklist = set()
    for k, history in grid_history.items():
        if history['max_streak'] > CONSECUTIVE_THRESHOLD:
            # 안전장치: 변동성 체크
            w_std = statistics.stdev(history['w_list']) if len(history['w_list']) > 1 else 0
            h_std = statistics.stdev(history['h_list']) if len(history['h_list']) > 1 else 0
            
            if w_std < STD_DEV_THRESHOLD and h_std < STD_DEV_THRESHOLD:
                blacklist.add(k)

    print(f"  - Found {len(blacklist)} static grid zones.")
    return blacklist

def apply_filter(df, blacklist):
    """
    Blacklist에 해당하는 Grid의 박스를 제거
    """
    if not blacklist:
        return df
    
    # 각 행에 대해 Grid Key 계산
    # (벡터화 연산을 위해 numpy 사용)
    cx = df['x'] + df['width'] / 2
    cy = df['y'] + df['height'] / 2
    
    gx = (cx // GRID_SIZE).astype(int)
    gy = (cy // GRID_SIZE).astype(int)
    
    # 필터링 마스크 생성
    # (gx, gy) 튜플을 만들어 비교하는 것은 판다스에서 느리므로, 문자열 키 등을 활용하거나 반복문 사용
    # 여기서는 apply를 사용하여 정확하게 처리
    def is_blacklisted(row):
        k = (int(row['gx']), int(row['gy']))
        return k in blacklist

    # 임시 데이터프레임 생성
    temp_df = pd.DataFrame({'gx': gx, 'gy': gy})
    mask = temp_df.apply(is_blacklisted, axis=1)
    
    # 반전시켜서 살릴 것만 남김
    return df[~mask].reset_index(drop=True)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files = list(INPUT_DIR.glob("*.parquet"))
    
    print(f"Found {len(parquet_files)} files in {INPUT_DIR}")
    
    for p_file in parquet_files:
        print(f"\nProcessing: {p_file.name} ...")
        try:
            df = pd.read_parquet(p_file)
            if df.empty:
                print("  - Empty file.")
                continue
                
            original_len = len(df)
            
            # 1. 블랙리스트 분석
            blacklist = get_static_blacklist(df)
            
            # 2. 필터링
            df_clean = apply_filter(df, blacklist)
            
            # 3. 저장
            cleaned_len = len(df_clean)
            removed = original_len - cleaned_len
            
            out_path = OUTPUT_DIR / p_file.name
            df_clean.to_parquet(out_path, index=False)
            
            print(f"  - Saved to {out_path}")
            print(f"  - Removed {removed} boxes ({removed/original_len*100:.1f}%)")
            
        except Exception as e:
            print(f"  - Error: {e}")

if __name__ == "__main__":
    main()