#written by Claudia Sagastizabal, v0 10/20, v1 01/21, GIT PNAS 05/21
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from sklearn.preprocessing import MinMaxScaler
import urllib.request as request
import os.path as path

def retrieve_data(url_data,download,csv_dir):
      head, tail = path.split(url_data)
      filename=path.join(csv_dir,tail)
      if download: request.urlretrieve(url_data, filename=filename)
      return filename

warnings.filterwarnings("ignore") # specify to ignore warning messages

my_path='./'
csv_dir=path.join(my_path,'CSV')
res_dir=path.join(my_path,'RES')
# Parameters
ddf = pd.read_csv(path.join(csv_dir,'Windows4AR.csv'), sep=';') #file with all the AR models to try
TAU=ddf.TAU[0] ; SUBNOT=ddf.SUBNOT[0]; WIN=7; 
#dados SP, from https://www.seade.gov.br/coronavirus/#
url_CASES="https://raw.githubusercontent.com/seade-R/dados-covid-sp/master/data/dados_covid_sp.csv"
download=False;fname_cases=retrieve_data(url_CASES,download,csv_dir)
url_BEDS="https://raw.githubusercontent.com/seade-R/dados-covid-sp/master/data/plano_sp_leitos_internacoes.csv"
download=False;fname_beds=retrieve_data(url_BEDS,download,csv_dir)
print('processing ',fname_cases,' and ',fname_beds)
 
country=0; Names=['SaoPaulo','Germany','UK'];codes=['SP','GE','UK']
Capacity=[-99,18000,4480]; pop=[46.29,83.78,67.89]; pref=codes[country]
#: nome_munic	codigo_ibge	dia	mes	datahora	casos	casos_novos	casos_pc	casos_mm7d	obitos	obitos_novos	obitos_pc	obitos_mm7d	letalidade	nome_ra	cod_ra	nome_drs	cod_drs	pop	pop_60	area	map_leg	map_leg_s	latitude	longitude	semana_epidem
df = pd.read_csv(fname_cases, sep=';', decimal=",", parse_dates=['datahora'])
df.dropna()
df=df.groupby('datahora').sum()
dfcasos_novos=df['casos_novos']
#datahora;nome_drs;pacientes_uti_mm7d;total_covid_uti_mm7d;ocupacao_leitos;pop;leitos_pc;internacoes_7d;internacoes_7d_l;internacoes_7v7
df = pd.read_csv(fname_beds, sep=';', decimal=",", parse_dates=['datahora'])
df.dropna(); df.index=df.datahora
df=df[df.nome_drs=='Estado de São Paulo']
col_hist=['Cap-'+pref, 'Cases-'+pref,  'ICU-'+pref]
ts=pd.DataFrame(index=df.index)
ts[col_hist[0]]=int(df['total_covid_uti_mm7d'].mean())
ts[col_hist[1]]=dfcasos_novos.rolling(WIN,min_periods=1).mean()
ts[col_hist[2]]=df['pacientes_uti_mm7d'].rolling(WIN,min_periods=1).mean()
col_new=['Corrected-'+pref,'Accum Corrected-'+pref,'Ratio-'+pref]
ts[col_new[0]]= [float(x)*SUBNOT for x in ts[col_hist[1]]]
ts[col_new[1]]=ts[col_new[0]].rolling(WIN,min_periods=1).sum().values  #accumulates days 
num=ts[col_hist[2]].rolling(WIN,min_periods=1).mean()
den=ts[col_new[1]].values
ts[col_new[2]] =num/den 

filename=path.join(res_dir,'HistoryRatioCasesCapacity.csv')
ts.to_csv(filename, sep=';')  #for tendency
tsdate=ts.index
last=tsdate.max();

#p,d,q,ct,Name,year_start,month_start,day_start,year_end,month_end,day_end
for rg in np.arange(0,len(ddf)):
    srg=ddf['Name'][rg]
    first=datetime(ddf.year_start[rg],ddf.month_start[rg],ddf.day_start[rg])
    start=max(first,datetime(ddf.year_start[rg],ddf.month_start[rg],ddf.day_start[rg]))
    end =min(last,datetime(ddf.year_end[rg],ddf.month_end[rg],ddf.day_end[rg]))
    mask_begin = tsdate >= start
    tts=ts[mask_begin][WIN+1:] #because of accumulated cases over WIN, the first WIN records are not good
    ToFit=pd.DataFrame(tts[col_new[2]]) #Ratio
    #scale
    sc_in = MinMaxScaler(feature_range=(0, 1))
    scaled_input = sc_in.fit_transform(ToFit);
    scaled_input =pd.DataFrame(scaled_input)
    X= scaled_input #
    sc_out = MinMaxScaler(feature_range=(0, 1))
    scaler_output = sc_out.fit_transform(ToFit)
    scaler_output =pd.DataFrame(scaler_output)
    y=scaler_output
    #test from end to last dates
    X.rename(columns={0:'Scaled Ratio'}, inplace=True)
    X.index=ToFit.index
    y.rename(columns={0:'Ratio'}, inplace= True)
    y.index=ToFit.index
    mask_train = (X.index >= start) & (X.index <= end)
    train_X=  X[mask_train]; train_y = y[mask_train]
    mask_test  = (X.index > end)
    test_X=  X[mask_test]; test_y = y[mask_test]
    train_size=len(train_X)
    test_size=len(test_X)
    #output file with callibration
    filename=path.join(res_dir, pref+'ARdata'+srg+'.csv')
    df=pd.DataFrame();
    param=(ddf.p[rg],ddf.d[rg],ddf.q[rg]);
    trend=ddf.ct[rg] 
    minm=[]
    minm=[sc_in.data_min_, sc_in.data_max_]
    df['Min']=minm[0]
    df['Max']=minm[1]
        
    mod = SARIMAX(train_y, exogenous=train_X,
                    order=param, 
                    trend=trend,
                    measurement_error=True, 
                    enforce_stationarity=True, 
                    enforce_invertibility=False)
    results = mod.fit(disp=0)
    #print(results.summary())
    sarall=results.params
    for s in range(len(sarall)):
                col=sarall.index[s]
                df[col]=sarall[col] 

    df['first record']=start
    df['last record']=end
    df['SUBNOT']=SUBNOT
    df['tau']=TAU
    df['WIN']=WIN
    df.to_csv(filename,sep=';', index=False, header=True, float_format="%.8f")
    print('done: ',filename)
