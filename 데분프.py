import pandas as pd  
import numpy as np 
import geopandas as gpd  
from shapely.geometry import Point
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans

#데이터 불러오기
data_path = "C:/Users/minin/OneDrive/바탕 화면/교통사고정보.csv"
data = pd.read_csv(data_path, encoding='cp949') 

#데이터 전처리
data = data.drop(columns=['발생년','발생년월일시','사고유형_중분류','도로형태','도로형태_대분류','가해자_당사자종별','피해자_당사자종별','발생위치X','발생위치Y'])

data['총피해자수'] = data[['사망자수', '부상자수', '중상자수', '경상자수', '부상신고자수']].sum(axis=1)

print(data.head())

data.drop(columns=['사망자수', '부상자수', '중상자수', '경상자수', '부상신고자수'], inplace=True)

print(data.head())

# 주야별 사고 발생 건수
day_night_counts = data['주야'].value_counts()
print(day_night_counts)

# 한글 폰트가 깨지지 않도록 설정
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False 

# 시각화
day_night_counts.plot(kind='bar', color='skyblue', title='주야별 사고 발생 건수')
plt.xlabel('주야')
plt.ylabel('사고 건수')
plt.show()

# 요일별 사고 발생 건수
weekday_counts = data['요일'].value_counts()
print(weekday_counts)

weekday_counts.plot(kind='bar', color='salmon', title='요일별 사고 발생 건수')
plt.xlabel('요일')
plt.ylabel('사고 건수')
plt.show()

# 강원도 지역으로 범위 줄이기
filtered_data = data[data['발생지시도'].str.startswith('강원')]
print(filtered_data.head())

# 사고 발생 위치의 위도, 경도 정보만 추출
locations_with = filtered_data[['위도', '경도','총피해자수']]

# K-means 클러스터링 적용
kmeans = KMeans(n_clusters=7, random_state=42)  # 클러스터 수는 적절히 조정 (5개 클러스터 예시)
filtered_data['클러스터'] = kmeans.fit_predict(locations_with)

# 클러스터의 중심 좌표
centroids = kmeans.cluster_centers_

# 클러스터링 결과 확인
print(filtered_data[['위도', '경도','총피해자수', '클러스터']].head())

# GeoDataFrame으로 변환 (위도, 경도 기반)
geometry = [Point(xy) for xy in zip(filtered_data['경도'], filtered_data['위도'])]
geo_data = gpd.GeoDataFrame(filtered_data, geometry=geometry)

# 좌표계 설정 (WGS84)
geo_data.crs = "EPSG:4326"

# 사고 발생 위치와 클러스터 중심 시각화
fig, ax = plt.subplots(figsize=(10, 8))
base_map = geo_data.plot(ax=ax, color='lightgray', alpha=0.5)

# 클러스터별 사고 위치 점 표시
geo_data.plot(ax=base_map, column='클러스터', cmap='viridis', markersize=20, legend=True)

# 클러스터 중심 좌표 표시
for i, center in enumerate(centroids):
    ax.scatter(center[1], center[0], color='red', marker='X', s=200, label=f'Cluster {i+1} Center')

plt.title('강원도 내 표지판 최적 설치 지역 (K-means 클러스터링)')
plt.xlabel('경도')
plt.ylabel('위도')
plt.legend()
plt.show()




