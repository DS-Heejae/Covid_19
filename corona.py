#!/usr/bin/env python
# coding: utf-8

# ### 1. train 데이터 가져오기

# In[1]:


train = pd.read_csv('datas/datas(kaggle)/train.csv', encoding='utf-8', parse_dates=['Date'])


# In[2]:


train.info()


# In[3]:


train.isnull().sum()


# In[4]:


# Province/State 결측치가 많으므로, 제거
train.dropna(axis=1, inplace=True)

#데이터 결합 위해 미리 column 명 변경
train.rename(columns = {'Country/Region':'country', 'Date':'date', 'Province/State' :'province', 'Lat':'lat', 'Long':'long'}, inplace = True)
train


# ### 2. 온도 데이터 가져오기

# In[5]:


temperature = pd.read_csv("datas/datas(kaggle)/temperature_dataframe.csv", encoding='utf-8',parse_dates=[6])


# In[6]:


temperature.isnull().sum()


# In[7]:


# 결측치 많은 province 열과 불필요한 열 제거
temperature.drop(columns = ['index', 'id', 'province', 'cases', 'fatalities', 'capital'], inplace=True)
temperature


# ### 3. train 데이터와 temperature 데이터 결합

# In[8]:


df1 = pd.merge(train, temperature, how='left', on = ['country', 'date', 'lat', 'long'])
df1


# ### 4. 국가별 인구 데이터 가져오기

# In[9]:


world_population = pd.read_csv("datas/datas(kaggle)/population_by_country_2020.csv", encoding='utf-8')
world_population.head()


# In[10]:


world_population.info()


# In[11]:


world_population.rename(columns = {'Country (or dependency)': 'country','Density (P/Km²)': 'Density', 'Land Area (Km²)':'LandArea', 'Med. Age':'Med_Age',
                                   'Population (2020)': 'Population','Net Change': 'Net_Change' , 'Yearly Change': 'Yearly_Change'}, inplace = True)


# In[12]:


world_population


# ### 5. train, temperature, world_population 데이터 결합

# In[13]:


df2 = pd.merge(df1, world_population, how='left', on = 'country')
df2


# ### 6. EDA

# ### (1) 데이터 확인, 전처리

# In[14]:


# 결측치 0 대체, % 제거 후 float 형변환
df2['humidity'].fillna(0, inplace=True)
df2['sunHour'].fillna(0, inplace=True)
df2['tempC'].fillna(0, inplace=True)
df2['windspeedKmph'].fillna(0, inplace=True)

df2['Yearly_Change'].replace('N.A.','0', inplace=True)
df2['Yearly_Change']=df2['Yearly_Change'].replace('%','', regex=True).astype(float)

df2['Urban Pop %'].replace('N.A.','0', inplace=True)
df2['Urban Pop %'] = df2['Urban Pop %'].replace('%','', regex=True).astype(float)

df2['World Share'].replace('N.A.','0', inplace=True)
df2['World Share'] = df2['World Share'].replace('%','', regex=True).astype(float)

df2['Fert. Rate'].replace('N.A.','0', inplace=True)
df2['Fert. Rate'] = df2['Fert. Rate'].astype(float)

df2['Med_Age'].replace('N.A.','0', inplace=True)
df2['Med_Age'] = df2['Med_Age'].astype(float)

df2['Migrants (net)'].fillna(0, inplace=True)


# In[15]:


# world_population 정보 없는 국가: 제거
df2.dropna(how="any", inplace=True)


# In[16]:


df2.isnull().sum()


# In[17]:


df2.describe()


# ### (2) 국가별 확진자 수

# In[18]:


import chart_studio.plotly as py
import cufflinks as cf 
cf.go_offline(connected=True)

confiremd_num = pd.DataFrame(df2.ConfirmedCases.groupby(df2.country).agg(max))
confiremd_num.iplot(kind='bar')


# ### (3) 국가별 사망자 수

# In[19]:


death_num = pd.DataFrame(df2.Fatalities.groupby(df2.country).agg(max))
death_num.iplot(kind='bar')


# ### (4) 지도 상 시각화(국가별 사망자 수)

# In[20]:


import folium

map_df = df2[df2['date']=='2020-03-24'][['country','lat', 'long','ConfirmedCases', 'Fatalities']]
map_osm = folium.Map(location = [41.8905,12.4942], zoom_start = 3)


# In[21]:


for item in map_df.index:
    lat = map_df.loc[item, 'lat']
    long = map_df.loc[item, 'long']
    folium.CircleMarker([lat,long],
                        radius=map_df.loc[item, 'Fatalities']/200,
                        popup=map_df.loc[item,'Fatalities'],
                        color='orange',
                        fill=True).add_to(map_osm)
map_osm


# ### (5) 시간에 따른 누적 확진자, 사망자 수 증가 추이

# In[22]:


date_confirmed = pd.DataFrame(df2.ConfirmedCases.groupby(df2.date).agg(sum))
date_death = pd.DataFrame(df2.Fatalities.groupby(df2.date).agg(sum))


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure(figsize=(20,10))

plt.title('누적 확진자, 사망자 수', fontsize=20)

plt.plot(date_confirmed.ConfirmedCases, 'k', label='확진자 수')
plt.plot(date_death.Fatalities, 'r', label='사망자 수')

plt.legend(fontsize=17)
plt.show()


# ### 시계열 분석으로 확진자, 사망자 추이 예측

# #### 확진자 수

# In[24]:


date_ = pd.DataFrame(df2.ConfirmedCases.groupby(df2.date).agg(sum))
index_ = pd.DataFrame(date_.index)
confirmedcases_=pd.DataFrame(date_['ConfirmedCases'].values)
confirmedcases_df = pd.concat([index_, confirmedcases_], axis=1)
confirmedcases_df.columns = ['date','confirmedcases']
confirmedcases_df


# In[25]:


from fbprophet import Prophet
confirmedcase_df1 = pd.DataFrame({'ds': confirmedcases_df['date'], 'y':confirmedcases_df['confirmedcases']})
m = Prophet(daily_seasonality=True)
m.fit(confirmedcase_df1)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)
m.plot(forecast);


# In[26]:


m.plot_components(forecast);


# #### 사망자 수

# In[27]:


date_1 = pd.DataFrame(df2.Fatalities.groupby(df2.date).agg(sum))
index_1 = pd.DataFrame(date_1.index)
fatalities_=pd.DataFrame(date_1['Fatalities'].values)
fatalities_df = pd.concat([index_1, fatalities_], axis=1)
fatalities_df.columns = ['date','fatalities']
fatalities_df


# In[28]:


from fbprophet import Prophet

fatalities_df = pd.DataFrame({'ds': fatalities_df['date'], 'y':fatalities_df['fatalities']})
m = Prophet(daily_seasonality=True)
m.fit(fatalities_df)
future = m.make_future_dataframe(periods=30)
forecast1 = m.predict(future)
m.plot(forecast1);


# In[29]:


m.plot_components(forecast1);


# ### (6) 변수 간 상관 관계 파악

# In[30]:


import seaborn as sns
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = df2.corr()

fig, ax = plt.subplots( figsize=(20,15) )

mask = np.zeros_like(df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(df, 
            cmap = 'RdYlBu_r', 
            annot = True,   
            mask=mask,     
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            vmin = -1,vmax = 1 
           )  
plt.show()


# ### (7) 다중 공선성 확인

# In[31]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

df3 = df2.drop(columns = ['country', 'date']) #정수, 실수형 데이터만 남기기

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df3.values, i) for i in range(df3.shape[1])]
vif["features"] = df3.columns
vif = vif.sort_values("VIF Factor").reset_index(drop=True)
vif


# ### 7. OLS

# ### (1) train, test 데이터 구분, train 데이터에서 회귀 모형 만들기

# In[32]:


new_df2 = df2.drop(columns=['Fatalities'])
dfX = pd.DataFrame(new_df2, columns=new_df2.columns)
dfy = pd.DataFrame(df2.Fatalities, columns=["Fatalities"])
df = pd.concat([dfX, dfy], axis=1)


# In[33]:


N = len(df2)
ratio = 0.7
np.random.seed(0)
idx_train = np.random.choice(np.arange(N), np.int(ratio * N))
idx_test = list(set(np.arange(N)).difference(idx_train))

df_train = df2.iloc[idx_train]
df_test = df2.iloc[idx_test]


# In[34]:


import statsmodels.api as sm
model = sm.OLS.from_formula("Fatalities ~  scale(Population) + scale(windspeedKmph) + scale(ConfirmedCases)", df_train)
result = model.fit()
print(result.summary(()))


# ### (2) CCPR 

# In[35]:


fig = plt.figure(figsize=(10,15))
sm.graphics.plot_ccpr_grid(result, fig=fig)
fig.suptitle("")
plt.show()


# ### (3) test 데이터에서도 성능 구하기

# In[36]:


pred = result.predict(df_test)

rss = ((df_test.Fatalities - pred) ** 2).sum()
tss = ((df_test.Fatalities - df_test.Fatalities.mean())** 2).sum()
rsquared = 1 - rss / tss
rsquared


# ### (4) K폴드 교차검증

# In[37]:


from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.formula.api as smf
import statsmodels.api as sm

class StatsmodelsOLS(BaseEstimator, RegressorMixin):
    def __init__(self, formula):
        self.formula = formula
        self.model = None
        self.data = None
        self.result = None
        
    def fit(self, dfX, dfy):
        self.data = pd.concat([dfX, dfy], axis=1)
        self.model = smf.ols(self.formula, data=self.data)
        self.result = self.model.fit()
        
    def predict(self, new_data):
        return self.result.predict(new_data)


# In[38]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

model = StatsmodelsOLS("Fatalities ~  scale(Population) + scale(windspeedKmph) + scale(ConfirmedCases)")
cv = KFold(5, shuffle=True, random_state=0)
cross_val_score(model, dfX, dfy, scoring="r2", cv=cv)


# In[39]:


pred_Fatalities=pd.DataFrame(result.predict(dfX))
result_df = pd.concat([df2.Fatalities,pred_Fatalities], axis=1, names=['Fatalities', 'pred_Fatalities'])
result_df.rename(columns={0:'pred_Fatalities'}, inplace=True)
result_df

