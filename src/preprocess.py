import pandas as pd
import numpy as np

def load_and_preprocess_data():
    # 데이터 로드
    df_sessions = pd.read_csv('data/ev_charging_patterns.csv')
    df_ev = pd.read_parquet('data/ev.parquet')
    
    # 컬럼명 정규화 (Dataset A)
    # User ID,Vehicle Model,Battery Capacity (kWh),Charging Station ID,Charging Station Location,
    # Charging Start Time,Charging End Time,Energy Consumed (kWh),Charging Rate (kW),Charging Cost (USD),
    # State of Charge (Start %),State of Charge (End %),Temperature (°C),Vehicle Age (years),Charger Type,User Type
    
    col_mapping = {
        'User ID': 'User_ID',
        'Vehicle Model': 'Vehicle_Model',
        'Charging Station ID': 'Station_ID',
        'Charging Station Location': 'Station_Location',
        'Charging Start Time': 'Start_Time',
        'Charging End Time': 'End_Time',
        'Energy Consumed (kWh)': 'Energy_Consumed',
        'Charging Rate (kW)': 'Charging_Rate',
        'Charging Cost (USD)': 'Charging_Cost',
        'State of Charge (Start %)': 'State_of_Charge_Start',
        'State of Charge (End %)': 'State_of_Charge_End',
        'Temperature (°C)': 'Temperature',
        'Vehicle Age (years)': 'Vehicle_Age',
        'User Type': 'User_Type'
    }
    df_sessions = df_sessions.rename(columns=col_mapping)
    
    # 시간 데이터 변환
    df_sessions['Start_Time'] = pd.to_datetime(df_sessions['Start_Time'])
    df_sessions['End_Time'] = pd.to_datetime(df_sessions['End_Time'])
    df_sessions['hour'] = df_sessions['Start_Time'].dt.hour
    df_sessions['weekday'] = df_sessions['Start_Time'].dt.weekday # 0=월, 6=일
    
    # [1] 파생 컬럼 생성 로직
    df_sessions['session_duration_min'] = (df_sessions['End_Time'] - df_sessions['Start_Time']).dt.total_seconds() / 60
    # 완료 조건: 에너지 5kWh 이상 AND 종료 SoC 80% 이상
    df_sessions['is_completed'] = (df_sessions['Energy_Consumed'] >= 5) & (df_sessions['State_of_Charge_End'] >= 80)
    # 결제 조건: 비용 > 0
    df_sessions['is_paid'] = df_sessions['Charging_Cost'] > 0
    
    # 퍼널 단계 설정
    df_sessions['funnel_stage'] = '완료'
    df_sessions.loc[~df_sessions['is_paid'], 'funnel_stage'] = 'D3_결제미완'
    df_sessions.loc[~df_sessions['is_completed'], 'funnel_stage'] = 'D2_충전미완'
    
    # 이탈 사유 분류
    df_sessions['drop_reason'] = None
    df_sessions.loc[df_sessions['Energy_Consumed'] < 5, 'drop_reason'] = '충전량 부족(점유 이탈)'
    df_sessions.loc[df_sessions['session_duration_min'] < 5, 'drop_reason'] = '단시간 이탈(혼잡 이탈)'
    df_sessions.loc[df_sessions['Charging_Cost'] == 0, 'drop_reason'] = '결제 미완료'
    
    return df_sessions, df_ev

def detect_churn_signals(df):
    """
    User ID별로 Churn Signal 탐지
    """
    user_stats = df.groupby('User_ID').apply(lambda x: pd.Series({
        'recent_interval': x.sort_values('Start_Time')['Start_Time'].diff().dt.days.tail(5).mean(),
        'prev_interval': x.sort_values('Start_Time')['Start_Time'].diff().dt.days.head(5).mean(),
        'recent_energy': x.tail(5)['Energy_Consumed'].mean(),
        'baseline_energy': x['Energy_Consumed'].mean(),
        'station_switches': x['Station_ID'].nunique(),
    }), include_groups=False)
    
    # 위험 조건 및 점수 계산 (0-3점)
    cond1 = (user_stats['recent_interval'] > user_stats['prev_interval'] * 1.5)
    cond2 = (user_stats['recent_energy'] < user_stats['baseline_energy'] * 0.7)
    cond3 = (user_stats['station_switches'] > 3)
    
    user_stats['churn_risk_score'] = cond1.astype(int) + cond2.astype(int) + cond3.astype(int)
    user_stats['churn_risk'] = user_stats['churn_risk_score'] > 0
    
    # 원인 레이블링
    reasons = []
    for idx, row in user_stats.iterrows():
        r = []
        if cond1.loc[idx]: r.append('방문 간격 급증')
        if cond2.loc[idx]: r.append('충전량 급감')
        if cond3.loc[idx]: r.append('충전소 잦은 변경')
        reasons.append(', '.join(r) if r else '정상')
    user_stats['risk_reason'] = reasons
    
    return user_stats

if __name__ == "__main__":
    df_s, df_e = load_and_preprocess_data()
    print("Sessions Data Columns:", df_s.columns.tolist())
    churn = detect_churn_signals(df_s)
    print("Churn Detection Sample:\n", churn.head())
