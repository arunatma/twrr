import re
import sqlite3
import pandas as pd
import streamlit as st
import plotly_express as px
from pandas.tseries.offsets import MonthEnd
import calendar
from collections import deque
from datetime import datetime, date
import math
import base64
import requests
from io import StringIO


def get_download_link(filename, filenamelong, filetype):
    # Read a CSV file (replace with your actual file path)
    df = pd.read_csv(filename)
    # Convert to CSV (string) for download
    csv_data = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filenamelong}</a>'    
    return href

def highlight_vals(val):
    if val > 0:
        return "background-color: lightgreen; color: green; font-size: 9pt; font-family: Arial Narrow"
    elif val < 0:
        return "background-color: lightsalmon; color: red; font-size: 9pt; font-family: Arial Narrow"
    else:
        return "background-color: white; color: white; font-size: 9pt; font-family: Arial Narrow"
    
def prod_across(row):
    month_names = [x for (x,y) in row.items()][:12]
    prod_value = 1
    for x in month_names:
        val = row[x]
        if math.isnan(val):
            val = 0
        prod_value = prod_value * (1 + val)
    ret_value = prod_value - 1
    return ret_value


def get_scheme_codes():
    url = "https://portal.amfiindia.com/spages/NAVAll.txt"
    
    # Fetch the data
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    
    # Read the content
    content = response.text
    
    # Process the data
    list_code = []
    lines = content.split('\n')
    
    for line in lines:
        words = line.strip().split(';')
        if len(words) > 5:
            list_code.append([words[i] for i in [0, 1, 3]])
    
    df_codes = pd.DataFrame(list_code)
    df_codes.columns = ['schemeCode', 'schemeISIN', 'schemeName']
    return df_codes
    
@st.cache_data
def get_nav(scheme_code = '122639'):
    mf_url = 'https://api.mfapi.in/mf/' + scheme_code
    with urllib.request.urlopen(mf_url) as url:
        data = json.load(url)

    df_navs = pd.DataFrame(data['data'])
    df_navs['date'] = pd.to_datetime(df_navs.date, format='%d-%m-%Y')
    df_navs['nav'] = df_navs['nav'].astype(float)
    df_navs = df_navs.sort_values(['date']).set_index(['date'])
    df_dates = pd.DataFrame(pd.date_range(start=df_navs.index.min(), end=df_navs.index.max() + MonthEnd(0)), columns=['date']).set_index(['date'])
    df_navs = df_navs.join(df_dates, how='outer').ffill().reset_index()
    return df_navs



# latest_month
latest_month = date.today().strftime("%Y-%m-%d")
latest_month = st.sidebar.date_input("Latest Month:", value=datetime.strptime(latest_month, "%Y-%m-%d"), 
    min_value=datetime.strptime("1999-12-31", "%Y-%m-%d"),
    max_value=datetime.strptime(latest_month, "%Y-%m-%d"))

latest_month_code = latest_month
if latest_month:
    # Calculate the last day of the month for the selected date
    year = latest_month.year
    month = latest_month.month
    last_day = calendar.monthrange(year, month)[1]
    latest_month = date(year, month, last_day)
    
    # latest_month_code = latest_month.replace('-', '')
    latest_month_code = latest_month.strftime("%Y-%m-%d")
    st.sidebar.write("Using the month end date: " + latest_month_code)

# Samples files for download
link1 = get_download_link("ledger_complete.csv", "Sample Ledger CSV File", "text/plain")
link2 = get_download_link("date_and_values.csv", "Sample Portfolio Values CSV File", "text/plain")
st.sidebar.markdown(link1, unsafe_allow_html=True)
st.sidebar.markdown(link2, unsafe_allow_html=True)

# Load the Leger Balances
ledger_file = st.sidebar.file_uploader("Upload the Ledger CSV file", type=["csv"])
if ledger_file is not None:
    df_ledger = pd.read_csv(ledger_file)
else:
    df_ledger = pd.read_csv("ledger_complete.csv")

df_ledger['posting_date'] = pd.to_datetime(df_ledger['posting_date'])

# Get Bank Transactions
df_bank_txns = df_ledger[(df_ledger['voucher_type'] == 'Bank Receipts') | (df_ledger['voucher_type'] == 'Bank Payments')].reset_index(drop=True)
df_bank_txns['amount'] = df_bank_txns['debit'] - df_bank_txns['credit']

# Create Date Series
df_dates = pd.DataFrame(pd.date_range(df_ledger['posting_date'].min(), df_ledger['posting_date'].max()))
df_dates.columns = ['posting_date']
df_dates = df_dates.set_index(['posting_date'])

# Merge and do the forward fills
df_ledger_vals = df_ledger[df_ledger['posting_date'].isnull() == False].set_index('posting_date')
df_ledger_vals = df_ledger_vals.join(df_dates, how='right')
df_ledger_vals = df_ledger_vals[['net_balance']].ffill().reset_index()

# Retain only the month ends
df_month_ends = pd.DataFrame(pd.date_range('2000-07-31', latest_month_code, freq='M') + MonthEnd(1))
df_month_ends.columns = ['posting_date']
df_ledgers = df_month_ends.merge(df_ledger_vals, on='posting_date')
df_ledgers = df_ledgers.reset_index()

# Keep only the last value of the ledger on a given date
df_ledgers['rank'] = df_ledgers.reset_index().groupby(['posting_date'])['index'].rank(ascending=False)
df_ledgers = df_ledgers[df_ledgers['rank'] == 1]
del df_ledgers['index']
del df_ledgers['rank']

# Load Date and Values File - Captures the value of the portfolio on the month ends.
values_file = st.sidebar.file_uploader("Upload the Portfolio Values CSV file", type=["csv"])
if values_file is not None:
    df_date_vals = pd.read_csv(values_file)
else:
    df_date_vals = pd.read_csv('./date_and_values.csv')

# Retain only the month end values, merge with ledger to get the overall portfolio value
df_date_vals.columns = ['posting_date', 'value']
df_date_vals.posting_date = pd.to_datetime(df_date_vals.posting_date)
df_values = df_month_ends.merge(df_date_vals, on='posting_date')
df_values = df_values.merge(df_ledgers, on='posting_date')
df_values['total'] = df_values['value'] + df_values['net_balance']

# Selecting the related transactions from the accounts
df_date_txns = df_bank_txns[['posting_date', 'amount']].copy()
df_date_txns.posting_date = pd.to_datetime(df_date_txns.posting_date)

# Grouping several transactions happened on the same date
df_grp_txns = pd.DataFrame(df_date_txns.groupby('posting_date')['amount'].sum())
df_grp_txns = df_grp_txns.reset_index()

# Creating base_date and next_date for TWRR calculations
df_grp_txns['month_end'] = df_grp_txns['posting_date'] + MonthEnd(0)
df_grp_txns['base_date'] = df_grp_txns['posting_date'] + MonthEnd(-1)
df_grp_txns.loc[df_grp_txns['posting_date'] == df_grp_txns['month_end'], 'base_date'] = df_grp_txns['posting_date']
df_grp_txns['days'] = (df_grp_txns['posting_date'] - df_grp_txns['base_date']).dt.days
df_grp_txns['next_date'] = df_grp_txns['base_date'] + MonthEnd(1)
df_grp_txns['days_to_go'] = (df_grp_txns['next_date'] - df_grp_txns['posting_date']).dt.days + 1

df_cashflows = pd.DataFrame(df_grp_txns.groupby('base_date')['amount'].sum()).reset_index()
df_cashflows['amount'] = -1 * df_cashflows['amount']

df_grp_txns['weight'] = 1
df_grp_txns.loc[df_grp_txns['days'] != 0, 'weight'] = df_grp_txns['days_to_go'] / (df_grp_txns['days'] + df_grp_txns['days_to_go'] - 1)
df_grp_txns['weight'] = df_grp_txns['days_to_go'] / (df_grp_txns['days'] + df_grp_txns['days_to_go'] - 1)
df_grp_txns['wt_amount'] = df_grp_txns['weight'] * df_grp_txns['amount'] * -1

df_wt_vals = pd.DataFrame(df_grp_txns[['base_date', 'wt_amount']].groupby('base_date')['wt_amount'].sum()).reset_index()
df_month_ends.columns = ['base_date']
df_cfs = df_month_ends.merge(df_wt_vals.merge(df_cashflows, on='base_date', how='left'), on='base_date', how='left')
df_cfs = df_cfs.fillna(0)

df_portfolio = df_values[['posting_date', 'total']]
df_portfolio.columns = ['base_date', 'value']

df_all = df_portfolio.merge(df_cfs, on='base_date', how='left')
df_all['P1'] = df_all['value'].shift(-1)
df_all['P0'] = df_all['value']
df_all = df_all.fillna(0)

df_all['numerator'] = df_all['P1'] - df_all['P0'] - df_all['amount']
df_all['denominator'] = df_all['P0'] + df_all['wt_amount']
df_all['return'] = df_all['numerator'] / df_all['denominator']

df_abc = df_all.loc[1:].copy()
df_abc['multiple'] = df_abc['return'] + 1
df_abc['cumul_return'] = df_abc.multiple.cumprod()
df_abc['end_date'] = df_abc['base_date'] + MonthEnd(1)

df_returns = df_abc[['end_date', 'return', 'multiple', 'cumul_return']].copy()

first_row = pd.DataFrame({'end_date':pd.to_datetime(df_date_vals['posting_date'].iloc[0].strftime('%Y-%m-%d')),
                          'return':0, 'multiple':1, 'cumul_return':1}, index =[0])
df_returns_v1 = pd.concat([first_row, df_returns]).reset_index(drop = True)
df_returns_v1 = df_returns_v1[:-1]
df_returns_v1['ttm_return'] = df_returns_v1['multiple'].rolling(window=12).agg(lambda x: x.prod()) - 1

df_ret_long = df_returns.head(-1).copy()
df_ret_long['Month'] = df_ret_long['end_date'].dt.month
df_ret_long['Year'] = df_ret_long['end_date'].dt.year

df_ret_long['CY'] = 'CY' + df_ret_long['Year'].astype(str)

df_ret_long.loc[df_ret_long['Month'] <=3, 'FY'] = 'FY' + df_ret_long['Year'].astype(str)
df_ret_long.loc[df_ret_long['Month'] >=4, 'FY'] = 'FY' + (df_ret_long['Year'] + 1).astype(str)
df_ret_long['Month'] = pd.to_datetime(df_ret_long['Month'], format='%m').dt.month_name().str.slice(stop=3)

month_values = range(1,13)
month_names = [calendar.month_name[x][:3] for x in month_values]
month_names_fy = deque(month_names)
month_names_fy.rotate(-3)
month_names_fy = list(month_names_fy)

df_cy_returns = df_ret_long.pivot(index='CY', columns='Month', values='return')
df_cy_returns = df_cy_returns[month_names]

df_fy_returns = df_ret_long.pivot(index='FY', columns='Month', values='return')
df_fy_returns = df_fy_returns[month_names_fy]

df_cy_returns['CY'] = df_cy_returns.apply(prod_across, axis=1)
df_fy_returns['FY'] = df_fy_returns.apply(prod_across, axis=1)


#####################################
#             OUTPUT                #
#####################################

tab1, tab2, tab3 = st.tabs(["CAGR", "Monthly Returns", "Comparison"])

with tab1:
    # st.write("TWRR Monthly Table")
    df_returns_v1['Date'] = df_returns_v1['end_date'].astype(str)
    df_returns_v1['Return'] = round(df_returns_v1['return'] * 100, 2)
    
    fig_cumul = px.line(df_returns_v1, x='end_date', y='cumul_return', log_y=True, title="Cumulative Return")
    fig_cumul.update_layout(xaxis_title='Date', yaxis_title='Cumulative Return')
    st.write(fig_cumul)

    # CAGR
    df_returns_v1['years'] = (df_returns_v1['end_date'] - df_returns_v1['end_date'][0]).dt.days / 365.25
    df_returns_v1['cagr'] = (df_returns_v1['cumul_return'] ** (1 / df_returns_v1['years']) - 1) * 100
    fig_cagr = px.line(df_returns_v1[df_returns_v1['end_date']>='2018-03-31'], x='end_date', y='cagr', log_y=False)
    fig_cagr.update_layout(xaxis_title='Date', yaxis_title='Compounded Annual Growth Rate', title="CAGR")
    st.write(fig_cagr)

    # Trailing Twelve Month Returns
    # fig_ttm = px.line(df_returns_v1, x='end_date', y='ttm_return', log_y=False)
    # st.write(fig_ttm)

with tab2:
    st.write('Calendar Year')
    d = dict.fromkeys(df_cy_returns.select_dtypes('float').columns, "{:.2%}")
    cy_styled = df_cy_returns.style.applymap(highlight_vals).format(d)
    st.table(cy_styled)

    st.write('Financial Year')
    d = dict.fromkeys(df_fy_returns.select_dtypes('float').columns, "{:.2%}")
    fy_styled = df_fy_returns.style.applymap(highlight_vals).format(d)
    st.table(fy_styled)

with tab3:
    df_codes = get_scheme_codes()
    df_shortlist = df_codes[df_codes['schemeName'].str.contains(
        'flexi.*direct.*growth|flexi.*growth.*direct|momentum|nifty|mid|equity',
        flags=re.IGNORECASE, regex=True)]
    base_date = st.date_input("Base Date:", value=datetime.strptime("2020-03-31", "%Y-%m-%d"), 
                                        min_value=datetime.strptime("1999-12-31", "%Y-%m-%d"),
                                        max_value=datetime.strptime(latest_month_code, "%Y-%m-%d"))
    base_date = base_date + MonthEnd(0)
    st.write(base_date)
    scheme_names = st.multiselect("Select Schemes:", df_shortlist.schemeName)
    scheme_codes = list(df_shortlist[df_shortlist['schemeName'].isin(scheme_names)].schemeCode)
    
    df_returns_v2 = df_returns_v1.copy()
    mf_codes = scheme_codes
    mf_names = []
    for mf_code in mf_codes:
        mf_name = df_codes.loc[df_codes['schemeCode'] == mf_code, 'schemeName'].values[0]
        mf_names.append(mf_name)
        df_navs = get_nav(mf_code)
        df_navs.columns = ['end_date', mf_name]
        df_returns_v2 = df_returns_v2.merge(df_navs, on='end_date')    
    
    df_returns_v3 = df_returns_v2[df_returns_v2['end_date'] >= base_date].reset_index(drop=True)
    df_returns_v3['EquityPortfolio'] = df_returns_v3['cumul_return'] / df_returns_v3.loc[0, 'cumul_return']
    for mf_name in mf_names:
        df_returns_v3[mf_name] = df_returns_v3[mf_name] / df_returns_v3.loc[0, mf_name]

    df_ret_long = pd.melt(df_returns_v3, id_vars='end_date', 
        value_vars = mf_names + ['EquityPortfolio'], var_name='Portfolio', value_name='NAV')
    
    fig = px.line(df_ret_long,
                  x='end_date', y='NAV',
                  color_discrete_sequence=px.colors.qualitative.G10,
                  color='Portfolio',
                  log_y=True,
                  title='Performance Comparison: (Base Date: ' + base_date.strftime('%Y-%m-%d') + ')')
    fig.update_layout(legend=dict(yanchor="bottom", y=-0.5, xanchor="left", x=0))
    fig.update_layout(xaxis_title='Date', yaxis_title='Cumulative Return', height=700)
    st.write(fig)
