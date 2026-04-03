import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from preprocess import load_and_preprocess_data, detect_churn_signals
# user_clustering으로 이름 변경하여 캐시 회피
from user_clustering import get_user_features, run_clustering

# 페이지 설정
st.set_page_config(page_title="EV Infra Analytics Dashboard", layout="wide")

# 가상 좌표 데이터 (시각화용)
CITY_COORDS = {
    'Houston': [29.7604, -95.3698],
    'San Francisco': [37.7749, -122.4194],
    'Los Angeles': [34.0522, -118.2437],
    'Chicago': [41.8781, -87.6298],
    'New York': [40.7128, -74.0060]
}

@st.cache_data
def get_processed_data(refresh_key=0):
    df_s, df_e = load_and_preprocess_data()
    churn_df = detect_churn_signals(df_s)
    u_feat = get_user_features(df_s)
    clustered_df, pca_var = run_clustering(u_feat)
    return df_s, df_e, churn_df, clustered_df, pca_var

# 강제 캐시 갱신
df_s, df_e, churn_df, clustered_df, pca_var = get_processed_data(refresh_key=2)

# 사이드바
st.sidebar.title("⚡ EV Infra Dashboard")
view_mode = st.sidebar.radio("패널 선택", ["B2B 관리자 분석", "B2C 유저 맞춤 서비스"])

if view_mode == "B2B 관리자 분석":
    st.title("🏢 B2B 관리자 분석 패널")
    
    tab1, tab2, tab3 = st.tabs(["주요 KPI 및 맵", "EDA 테마별 분석", "유저 세그먼트"])
    
    with tab1:
        # KPI 카드
        col1, col2, col3, col4, col5 = st.columns(5)
        total_ev = len(df_e)
        total_stations = df_s['Station_ID'].nunique()
        
        with col1: st.metric("EV-to-Charger Ratio", f"{total_ev/total_stations:.1f}", delta="-1.5")
        with col2: st.metric("Station Density", "0.85/km²", delta="0.05")
        with col3: st.metric("Infra Gap Score", "74", delta="8", delta_color="inverse")
        with col4: st.metric("Utilization Rate", "32.4%", delta="2.1%")
        with col5: st.metric("Avg. Duration", "142m", delta="-8m")
        
        col6, col7, col8, col9, col10 = st.columns(5)
        with col6: st.metric("Peak Concentration", "28.5%", delta="1.2%")
        with col7: st.metric("Revenue per Session", f"${df_s['Charging_Cost'].mean():.1f}", delta="$0.5")
        with col8: st.metric("Churn Rate", "8.2%", delta="-0.4%")
        with col9: st.metric("Retention Rate", "84.1%", delta="1.5%")
        with col10: st.metric("Price Elasticity", "-1.24", delta="0.1")
        
        st.divider()
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📍 입지 스코어링 및 충전소 분포")
            m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
            for city, coord in CITY_COORDS.items():
                count = len(df_s[df_s['Station_Location'] == city])
                folium.CircleMarker(
                    location=coord,
                    radius=count/10,
                    popup=f"{city}: {count} Sessions",
                    color='blue', fill=True
                ).add_to(m)
            st_folium(m, width=900, height=450)
            
        with c2:
            st.subheader("⚙️ 피크 분산 시뮬레이터")
            discount = st.slider("비피크 할인율 (%)", 0, 50, 20)
            target_shift = discount * 0.45
            st.write(f"예상 수요 분산량: **{target_shift:.1f}%**")
            if target_shift >= 20:
                st.success("🎯 목표 달성: 피크타임 20% 분산 가능")
            else:
                st.warning("⚠️ 목표 미달: 추가 인센티브 필요")
            
            st.subheader("🎯 충전 전환 퍼널")
            fig_funnel = go.Figure(go.Funnel(
                y=['S1_탐색', 'S2_선택', 'S3_플러그인', 'S4_완료', 'S5_결제'],
                x=[2500, 2100, 1800, 1400, 1320],
                textinfo="value+percent initial"
            ))
            st.plotly_chart(fig_funnel, use_container_width=True)

    with tab2:
        st.header("🔍 EDA 5개 테마 상세 분석")
        
        # 1. 공간 분석
        st.subheader("1. 공간 분석: 지역별 차량 등록 밀도")
        city_counts = df_e['City'].value_counts().head(10).reset_index()
        fig_geo = px.bar(city_counts, x='City', y='count', color='count', title="상위 10개 도시별 차량 등록 수")
        st.plotly_chart(fig_geo, use_container_width=True)
        
        # 2. 시계열 분석
        st.subheader("2. 시계열 분석: 요일별/시간대별 충전 수요")
        time_pivot = df_s.pivot_table(index='Day of Week', columns='hour', values='User_ID', aggfunc='count').fillna(0)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        time_pivot = time_pivot.reindex(days)
        fig_time = px.imshow(time_pivot, labels=dict(x="Hour", y="Weekday", color="Sessions"), color_continuous_scale="Viridis")
        st.plotly_chart(fig_time, use_container_width=True)
        
        # 3. 차량 특성 분석
        st.subheader("3. 차량 특성: 배터리 용량별 충전 시간 분석")
        df_s['Battery_Group'] = pd.qcut(df_s['Battery Capacity (kWh)'], 4, labels=['Q1(Low)', 'Q2', 'Q3', 'Q4(High)'])
        fig_vehicle = px.box(df_s, x='Battery_Group', y='session_duration_min', color='Battery_Group', title="배터리 용량 쿼타일별 충전 시간")
        st.plotly_chart(fig_vehicle, use_container_width=True)
        
        # 4. 환경 변수 분석
        st.subheader("4. 환경 변수: 기온에 따른 충전 효율(kWh/min)")
        # 0 나누기 및 결측치 방지
        df_s['efficiency'] = df_s['Energy_Consumed'] / df_s['session_duration_min'].replace(0, np.nan)
        df_s['efficiency'] = df_s['efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
        fig_env = px.scatter(df_s, x='Temperature', y='efficiency', trendline="ols", 
                             title="기온 vs 충전 효율 (OLS 회귀선)")
        st.plotly_chart(fig_env, use_container_width=True)
        
        # 5. 비용 분석
        st.subheader("5. 비용 분석: 요금제 계층별 수요 분포")
        df_s['cost_tier'] = pd.qcut(df_s['Charging_Cost'], 4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
        fig_cost = px.histogram(df_s, x='hour', color='cost_tier', barmode='group', title="요금 Tier별 충전 시작 시간대 분포")
        st.plotly_chart(fig_cost, use_container_width=True)

    with tab3:
        st.subheader("🧪 유저 세그먼테이션 (PCA + Clustering)")
        c_1, c_2 = st.columns([1, 2])
        with c_1:
            st.write(clustered_df['Cluster_Name'].value_counts())
            st.info(f"PCA 설명분산비: {pca_var:.2%}")
        with c_2:
            fig_pca = px.scatter(clustered_df, x='pca_1', y='pca_2', color='Cluster_Name', symbol='is_outlier',
                                 hover_data=['avg_energy', 'session_count'])
            st.plotly_chart(fig_pca, use_container_width=True)

else:
    st.title("👤 B2C 유저 맞춤 서비스")
    user_list = df_s['User_ID'].unique()
    selected_user = st.sidebar.selectbox("유저 선택", user_list)
    
    u_data = df_s[df_s['User_ID'] == selected_user].sort_values('Start_Time')
    u_cluster = clustered_df.loc[selected_user, 'Cluster_Name']
    
    st.header(f"안녕하세요, :blue[{selected_user}] 님!")
    st.subheader(f"회원님의 페르소나 유형은 **[{u_cluster}]** 입니다.")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("📊 나의 충전 리포트")
        fig_u_trend = px.area(u_data, x='Start_Time', y='Energy_Consumed', title="나의 충전량 트렌드")
        st.plotly_chart(fig_u_trend, use_container_width=True)
        
    with col_v2:
        st.subheader("💡 최적 충전 가이드")
        st.info("🕒 지금 추천하는 피크 회피 시간: **23:00 - 02:00**")
        st.metric("예상 절약 금액", "$12.5", delta="-$2.1", delta_color="normal")
        
    st.divider()
    st.subheader("🎁 에코 포인트 시뮬레이션")
    off_peak_sessions = st.slider("이번 달 비피크 충전 예상 횟수", 0, 20, 4)
    expected_points = off_peak_sessions * 1000
    st.write(f"예상 적립 포인트: **{expected_points:,} P**")
    
    if churn_df.loc[selected_user, 'churn_risk']:
        st.error(f"⚠️ **알림**: {churn_df.loc[selected_user, 'risk_reason']}으로 인해 이용 패턴이 변화되었습니다.")
        st.markdown("다시 찾아주시는 회원님을 위한 **특별 복귀 쿠폰(30% 할인)**을 확인하세요!")
        st.button("쿠폰 받기")
