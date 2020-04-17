# Covid_19

Kaggle Corona 데이터 EDA, 사망자 수 예측 간단한 선형회귀분석(OLS 이용)

1. 코드 확인:
https://nbviewer.jupyter.org/github/DS-Heejae/Covid_19/blob/master/corona.ipynb

2. 데이터 설명

- kaggle 데이터
  - train 데이터
    - Id,Province/State,Country/Region,Lat,Long,Date(2020-01-22~2020-03-24),ConfirmedCases(누적 확진자수),Fatalities(누적 사망자수)
    
  - test 데이터
    - ForecastId,Province/State,Country/Region,Lat,Long,Date
    
    
- 데이터 분석에 추가로 사용한 외부 데이터
  - 기후 데이터
    - 기온, 풍속, 습도 등 기후와 확진자/사망자 수 간의 상관 관계 파악 위해
    - 출처: https://www.kaggle.com/winterpierre91/covid19-global-weather-data by @winterpierre91

  - 2020 국가별 인구 데이터
    - 인구 수, 밀도 등 인구 데이터와 확진자/사망자 수 간의 상관 관계 파악 위해
    - 출처: https://www.kaggle.com/tanuprabhu/population-by-country-2020
