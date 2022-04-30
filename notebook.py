# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + language="javascript"
# var rate = 0;
# // apply setting to  all current CodeMirror instances
# IPython.notebook.get_cells().map(
#     function(c) {  return c.code_mirror.options.cursorBlinkRate=rate;  }
# );
# // make sure new CodeMirror instance also use this setting
# CodeMirror.defaults.cursorBlinkRate=rate;
# -

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display
from matplotlib.backends.backend_pdf import PdfPages
import os
from sklearn.preprocessing import StandardScaler
import pmdarima as pm

df = pd.read_excel('./data/Статистические_данные_показателей_СЭР_cut.xlsx', index_col=0).T
df_sc = pd.read_excel('./data/сценарка.xlsx', index_col=0).T

df_sc = df_sc.iloc[1:,:].loc[:,df_sc.columns.notna()].dropna(how='all').dropna(axis=1,how='all')

df = df.apply(pd.to_numeric)
df_sc = df_sc.apply(pd.to_numeric)

df_sc=df_sc.interpolate()
df=df.interpolate()

df.columns.values[23] += ' 1'
# df.columns.values[73] += ' 1'

# +
# df.iloc[:,69:80]

# +
# df_no_na = df.dropna(axis=1, thresh=12)

# +
# df.isnull().sum(axis = 0).sort_values(ascending=False)
# -

df.isnull().sum(axis = 0).value_counts().sort_index(ascending=False)

df.isnull().sum(axis = 1)

df_sc.isnull().sum(axis = 0).value_counts().sort_index(ascending=False)

df_sc.isnull().sum(axis = 1)

# df_no_idx = df[df.columns.drop(list(df.filter(regex='Индекс.*')))]
df_renamed = df.rename({col: ' '.join(el[:5].capitalize() if len(el)>1 else el for el in col.split(' '))[:60] for col in df.columns}, axis=1)

for i in range(int(df_renamed.shape[1]/10)+1):
    display(df_renamed.iloc[:,i*10:(i+1)*10])


# +
# for i in range(df_renamed.shape[1]):
#     _, ax = plt.subplots()
#     ax.plot(df_renamed.iloc[:,i])
#     ax.set_title(df_renamed.iloc[:,i].name)
# -

def print_stats(df_input, name, path=None):
    plt.style.use('seaborn')
    sns.set(font_scale=1)

    if path is not None:
        if not os.path.isdir(path):
            os.mkdir(path)
        name = path+'/'+name
    file_path = f'{name}_visualization.pdf'
    pp = PdfPages(file_path)
    plt.style.use('seaborn')
    sns.set(font_scale = 0.8)
    sns.set_style('whitegrid')

    for i in range(int(df_input.shape[1]/4)+1):
        num = i*4
        plt.figure(figsize=(11.69,8.27),dpi=200)
        for j in range(min(4, df_input.shape[1]-num)):
            plt.subplot(2, 2, 1+j)
            plt.xticks(np.arange(int(df_input.index[0]/5)*5, int(df_input.index[-1]/5)*5+5, 5))
            sns.lineplot(x=df_input.index, y=df_input.iloc[:, num+j].name, data=df_input,  ci=None, color='g', marker='o')
        pp.savefig()
    pp.close()


# %%capture
print_stats(df_renamed, 'target')

# %%capture
print_stats(df_sc, 'features')

df_index = df_renamed.filter(regex='^Индек.*')
df_economic = df_renamed.filter(regex='^(Проду|Вало|Объем|Оборо).*') # Detrend
df_trade = df_renamed.filter(regex='^(Импор|Экспо|Ввод|Дефиц).*') # Log
df_finance = df_renamed.filter(regex='^(Инвес|Собст|Привл|Креди|Заемн|Бюдже).*') # Detrend
df_social = df_renamed.filter(regex='^(Прожи|Трудо|Пенси|Детей|Номин|Фонд).*') # Detrend
df_labor = df_renamed.filter(regex='^(Числе|Урове).*')
# df_budget = df_renamed.filter(regex='(.*Бюдже.*)') # Detrend + Log abs
sum(len(df.columns) for df in [df_index, df_economic, df_trade, df_finance, df_social, df_labor])
drop_columns = set(df.columns).difference(
    set(el for cols in [df_index, df_economic, df_trade, df_finance, df_social, df_labor] for el in cols))

set(col for col in df_renamed.columns
   if col not in
    set(el for cols in [df_index, df_economic, df_trade, df_finance, df_social, df_labor] for el in cols))

# %%capture
for df_i, name in zip([df_index, df_economic, df_trade, df_finance, df_social, df_labor],
                    ['Index', 'Economic', 'Trade', 'Finance', 'Social', 'Labor']):
    print_stats(df_i, name, 'figures')

for df_i in [df_index, df_trade, df_labor]:
    display(df_i.loc[:,df_i.max().values/df_i.min().values > 10].columns)

df_economic_dt = df_economic.diff(axis=0)
df_finance_dt = df_finance.diff(axis=0)
df_social_dt = df_social.diff(axis=0)

for df_i, name in zip([df_economic_dt, df_finance_dt, df_social_dt],
                    ['Economic', 'Finance', 'Social']):
    print_stats(df_i, name+'_dt', 'figures/diff')

df_economic_dt


def print_correlations(df_input, name, path=None, fontsize=11, annot=True):
    if path is not None:
        if not os.path.isdir(path):
            os.mkdir(path)
        name = path+'/'+name
    pp = PdfPages(f'{name}_correlations.pdf')
    corr_mat = df_input.corr(method='pearson')
    sns.set(font_scale=1)
    plt.figure(figsize=(12.69, 8.27),dpi=100)
    plt.suptitle('correlations', fontsize=fontsize)
    hm = sns.heatmap(corr_mat, cbar=True, annot=annot, square=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,
                     annot_kws={'size': 8}, yticklabels=df_input.columns, xticklabels=df_input.columns)
    pp.savefig()
    pp.close()


def print_correlations_block(df_inputs, name, path=None, fontsize=11, annot=True):
    df_total = pd.concat([df for df in df_inputs], axis=1)
    if path is not None:
        if not os.path.isdir(path):
            os.mkdir(path)
        name = path+'/'+name
    pp = PdfPages(f'{name}_correlations.pdf')
    corr_mat = df_total.corr(method='pearson')
    sns.set(font_scale=1)
    plt.figure(figsize=(12.69, 8.27),dpi=100)
    plt.suptitle('correlations', fontsize=fontsize)
    ax = sns.heatmap(corr_mat, cbar=True, annot=annot, square=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,
                     annot_kws={'size': 8}, yticklabels=df_total.columns, xticklabels=df_total.columns)
    ax.hlines([sum(df.shape[1] for df in df_inputs[:i+1]) for i in range(len(df_inputs))], *ax.get_xlim(), color='black')
    ax.vlines([sum(df.shape[1] for df in df_inputs[:i+1]) for i in range(len(df_inputs))], *ax.get_ylim(), color='black')
    pp.savefig()
    pp.close()


# %%capture
df_total = pd.concat([df_index, df_economic_dt, df_trade, df_finance_dt, df_social_dt, df_labor, df_sc], axis=1)
for df_i, name in zip([df_index, df_economic_dt, df_trade, df_finance_dt, df_social_dt, df_labor, df_sc],
                    ['Index', 'Economic_dt', 'Trade', 'Finance_dt', 'Social_dt', 'Labor', 'Scenario']):
    print_correlations(df_i, name, 'figures/correlations')

# %%capture
print_correlations_block([df_index, df_economic_dt, df_trade, df_finance_dt, df_social_dt, df_labor, df_sc], 'Total', 'figures/correlations', 6, False)

index_scaler = StandardScaler()
df_index_scaled = pd.DataFrame(index_scaler.fit_transform(df_index), index=df_index.index, columns=df_index.columns)

# %%capture
print_stats(df_index_scaled, 'Index_scaled', 'figures/processed')

series_index_0 = df_index.iloc[::-1,0][df_index.iloc[::-1,0].notnull()]
train = series_index_0[:int(0.7*len(series_index_0))]
valid = series_index_0[int(0.7*len(series_index_0)):]

# +
model = pm.auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)

forecast = model.predict(n_periods=len(valid))
df_forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])
# -

plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(df_forecast, label='Prediction')
plt.show()


