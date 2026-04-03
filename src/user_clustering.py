import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

def get_user_features(df):
    """
    유저별 클러스터링용 피처 생성
    """
    user_features = df.groupby('User_ID').agg(
        avg_energy=('Energy_Consumed', 'mean'),
        avg_duration=('session_duration_min', 'mean'),
        avg_rate=('Charging_Rate', 'mean'),
        session_count=('User_ID', 'count'),
        peak_hour_ratio=('hour', lambda x: (x.between(7, 9) | x.between(18, 20)).mean()),
        weekend_ratio=('weekday', lambda x: (x >= 5).mean()),
        station_diversity=('Station_ID', 'nunique'),
        avg_soc_start=('State_of_Charge_Start', 'mean'),
    )
    # 1. NaN 및 Inf를 0으로 전처리
    user_features = user_features.replace([np.inf, -np.inf], 0).fillna(0)
    return user_features

def run_clustering(user_features):
    # 2. 스케일링 전 데이터 전처리 재확인
    user_features = user_features.replace([np.inf, -np.inf], 0).fillna(0)
    
    # [STEP 1] StandardScaler 정규화
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(user_features)
    
    # 3. 스케일링 후 혹시 모를 NaN 발생(분산 0 등) 대응
    if np.any(np.isnan(scaled_features)):
        scaled_features = np.nan_to_num(scaled_features)
    
    # [STEP 2] PCA(n_components=3) 차원 축소
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_features)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    
    # [STEP 4] KMeans(n_clusters=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(pca_result)
    
    # [STEP 5] DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(pca_result)
    
    # 결과 통합
    results = user_features.copy()
    results['Cluster'] = kmeans_labels
    results['is_outlier'] = (dbscan_labels == -1)
    results['pca_1'] = pca_result[:, 0]
    results['pca_2'] = pca_result[:, 1]
    results['pca_3'] = pca_result[:, 2]
    
    cluster_names = {0: '출퇴근형', 1: '장거리 여정형', 2: '평일 알뜰형'}
    results['Cluster_Name'] = results['Cluster'].map(cluster_names)
    results.loc[results['is_outlier'], 'Cluster_Name'] = '불규칙형(노이즈)'
    
    return results, explained_variance
