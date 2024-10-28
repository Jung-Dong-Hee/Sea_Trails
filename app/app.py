import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import re
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection

plt.style.use('ggplot')

st.set_page_config(page_title="Sea Trials - Streamlit", layout="centered")

st.title("해상 시운전 비용 분석")

try:
    df = pd.read_csv('app/pre_test_fin.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('app/pre_test_fin.csv', encoding='cp949')

df['Date'] = pd.to_datetime(df['Date'], format='%Y년 %m월', errors='coerce')

translation_dict = {
    r'1차_1차선': '1st_1st',
    r'2차_1차선': '2nd_1st',
    r'1차선': '1st',
    r'시리즈': 'Series'
}

def translate_ship_type(name):
    for key, value in translation_dict.items():
        name = re.sub(key, value, name)
    return name

df['선종'] = df['선종'].apply(translate_ship_type)

st.sidebar.title("Filter Options")
st.sidebar.header("분석 기간 및 대상 선종 선택")
start_year = st.sidebar.selectbox("시작 연도 선택", [str(year) for year in range(2015, 2025)], index=8, key="start_year")
start_month = st.sidebar.selectbox("시작 월 선택", [str(month).zfill(2) for month in range(1, 13)], index=6, key="start_month")
end_year = st.sidebar.selectbox("종료 연도 선택", [str(year) for year in range(2015, 2025)], index=9, key="end_year")
end_month = st.sidebar.selectbox("종료 월 선택", [str(month).zfill(2) for month in range(1, 13)], index=6, key="end_month")
ship_type = st.sidebar.selectbox("선종 선택", df['선종'].unique(), key="ship_type")
start_date = f"{start_year}-{start_month}-01"
end_date = f"{end_year}-{end_month}-01"
df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['선종'] == ship_type)]

with st.expander("데이터 설명 보기"):
    st.write("**Date**: 시운전 진행 일자.")
    st.write("**지연 여부**: 시운전 중 지연 발생 여부. (정상 운영 / 지연 운영)")
    st.write("**업체 타입**: 시운전 시행 업체의 타입.  (Direct = 직영, Cooperation = 협력)")
    st.write("**선종**: 시운전을 실시한 선박의 유형.")
    st.write("**시운전 일수** : 시운전의 실제 기간. (단위: Days)")
    st.write("**HFO** : 시운전에서 사용한 HFO의 양. (단위: ℓ )")
    st.write("**MFO** : 시운전에서 사용한 MFO의 양. (단위: ℓ )")
    st.write("**선장_투입~노무_안전_투입**: 시운전에 참여한 해당 인원수. (단위: 명)")

st.write("### 데이터 테이블")
st.dataframe(df_filtered)
if df_filtered.empty:
    st.write("선택한 기간 및 선박 유형에 대해 사용 가능한 데이터가 없습니다.")
else:

    # 시운전 테스트 횟수 분석
    st.write("### 01. 월별 해상 시운전 테스트 횟수")
    st.write("정상 운영, 지연 운영 월별 시운전 횟수를 분석")
    with st.expander("정상/지연 항목별 비용 분석 설명 보기"):
        st.write("**Normal**: 해당 시운전이 계획된 기간에 맞게 진행이 되었음을 의미.")
        st.write("**Delayed**: 해당 시운전이 계획된 기간(days)보다 지연이 되었음을 의미.")

    df_filtered['Month'] = df_filtered['Date'].dt.to_period('M')
    status_counts = df_filtered.groupby(['Month', '지연 여부']).size().unstack(fill_value=0)

    normal_counts = status_counts.get('정상', pd.Series(0, index=status_counts.index))
    delayed_counts = status_counts.get('지연', pd.Series(0, index=status_counts.index))

    total_counts = normal_counts + delayed_counts

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(status_counts.index.astype(str), normal_counts, label='Normal', color='#1ca392')
    ax.bar(status_counts.index.astype(str), delayed_counts, bottom=normal_counts, label='Delay', color='#ffc81b')

    ax.set_xlabel('Month')
    ax.set_ylabel('Count')
    ax.set_title(f'Monthly Normal/Delay Operations ({start_date} ~ {end_date}, {ship_type})')
    ax.set_xticklabels(status_counts.index.astype(str), rotation=45)
    ax.legend()
    plt.tight_layout()

    st.pyplot(fig)

    with st.expander("시운전 테스트 횟수 보기"):
        table_counts = pd.DataFrame({
            "월": status_counts.index.astype(str),
            "Normal": normal_counts.values,
            "Delay": delayed_counts.values,
            "Total": total_counts.values
        })
    with st.expander("시운전 테스트 횟수 보기"):
            st.markdown("### 월별 시운전 테스트 횟수 (단위: 건)")
            st.table(table_counts)
    
    with st.expander("카운트 상세 보기"):
        count_data = {
            "상태": ["Normal", "Delay", "Total"],
            "카운트": [normal_counts.sum(), delayed_counts.sum(), total_counts.sum()]
        }
        df_counts = pd.DataFrame(count_data)
    with st.expander("카운트 상세 보기"):
            st.markdown("### 정상, 지연, 총합 카운트 (단위: 건)")
            st.table(count_data)

    # 항목별 비용 분석
    st.write("### 02. 항목별 비용 분석")
    st.write("선종의 정상 운영, 지연 운영, Total 운영의 유류비, 인건비, 기타 경비, 총 경비 분석")
    with st.expander("항목별 비용 분석 설명 보기"):
        st.write("#### 항목명")
        st.write("**Fuel Cost**: 시운전 선박이 사용한 유류비를 의미.")
        st.write("**STN_C**: Sea Trials Navigator Cost의 약자로 항해사비를 의미. 항해사비는 선장 비용, 타수 비용, 도선비, 임시항해 검사비, 자차 수정 비용이 포함된 금액.")
        st.write("**SMMT_C**: Ship Maintenance and Management Team의 약자로 노무비를 의미.")
        st.write("**Other Cost** : 시운전 선박의 기타 경비를 의미. 기타 비용은 용도품 침구 및 물품, 예선료, 통선비, 양식, 한식 이 포함된 금액.")
        st.write("**Total Cost**: 시운전 선박의 총경비를 의미.")
        st.write("#### 운영 결과")
        st.write("**Normal**: 해당 시운전이 계획된 기간에 맞게 진행이 되었음을 의미.")
        st.write("**Delay**: 해당 시운전이 계획된 기간(days)보다 지연이 되었음을 의미.")
        st.write("**Total**: 시운전 선박의 모든 데이터를 의미.")

    def convert_to_millions(value):
        return value / 1_000_000

    statuses = ['정상', '지연', '전체']
    status_labels = {'전체': 'Total', '정상': 'Normal', '지연': 'Delay'}

    cost_types = ['유류비(\)', '항해사비', '노무원비용', '기타비용', '총 경비']
    cost_type_labels = {'유류비(\)': 'Fuel Cost', '항해사비': 'STN_Cost', '노무원비용': 'SMMT_Cost' , '기타비용': 'Other Cost', '총 경비': 'Total Cost'}

    data = {}
    for status in statuses:
        if status == '전체':
            df_status_filtered = df_filtered
        else:
            df_status_filtered = df_filtered[df_filtered['지연 여부'] == status]
        
        avg_costs = df_status_filtered[cost_types].mean()
        data[status] = [convert_to_millions(avg_costs[cost_type]) for cost_type in cost_types]

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [cost_type_labels[cost_type] for cost_type in cost_types]
    x = np.arange(len(labels))
    width = 0.2

    for i, status in enumerate(statuses):
        ax.bar(x + i * width - width, data[status], width, label=status_labels[status], color={'정상': '#f15628', '지연': '#ffc81b', '전체': '#1ca392'}[status])

    ax.set_ylabel('Cost (in millions)')
    ax.set_title(f'Average Costs by Operation Status - {ship_type}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()

    st.pyplot(fig)

    with st.expander("비용 분석 결과 보기"):
        table_data = []
        for status in statuses:
            table_data.append({
            "상태": status_labels[status],
            "유류비": f"{data[status][0]:.2f}",
            "항해사비": f"{data[status][1]:.2f}",
            "노무원비용": f"{data[status][2]:.2f}",
            "기타비용": f"{data[status][3]:.2f}",
            "총 경비": f"{data[status][4]:.2f}"
        })

    df_table = pd.DataFrame(table_data)
    
    with st.expander("비용 분석 결과 보기"):
        st.markdown("### 비용 분석 결과 (단위: 백만 원)")
        st.table(df_table)

    # 정상/지연 항목별 비용 분석
    st.write("### 03. 정상/지연 항목별 비용 분석")
    st.write("선종의 정상 운영, 지연 운영, Total 운영의 총 경비, 인건비, 유류비를 시간의 흐름에 따른 분석")
    with st.expander("정상/지연 항목별 비용 분석 설명 보기"):
        st.write("#### 항목명")
        st.write("**Total Cost**: 시운전 선박의 총경비를 의미.")
        st.write("**Labor Cost**: 시운전에 사용된 항해사 및 노무원들의 인건비를 의미.")
        st.write("**Fuel Cost**: 시운전 선박이 사용한 유류비를 의미.")
        st.write("#### 운영 결과")
        st.write("**Total** : 시운전 선박의 모든 데이터를 의미.")
        st.write("**Normal**: 해당 시운전이 계획된 기간에 맞게 진행이 되었음을 의미.")
        st.write("**Delayed**: 해당 시운전이 계획된 기간(days)보다 지연이 되었음을 의미.")

    def create_labor_cost_column(df):
        df['인건비'] = df['선장비용'] + df['타수비용'] + df['노무원비용']
        return df

    df_filtered = create_labor_cost_column(df_filtered) 

    df_normal = df_filtered[df_filtered['지연 여부'] == '정상']
    df_delay = df_filtered[df_filtered['지연 여부'] == '지연']
    df_total = df_filtered

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_total.groupby('Date')['총 경비'].mean(), label='Total', linestyle='-', linewidth=2, color='#f15628')
    ax.plot(df_normal.groupby('Date')['총 경비'].mean(), label='Normal', linestyle='-', linewidth=2, color='#1ca392')
    ax.plot(df_delay.groupby('Date')['총 경비'].mean(), label='Delay', linestyle='-', linewidth=2, color='#ffc81b')
    ax.set_title(f'Total Cost for {ship_type} ({start_date} ~ {end_date})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total cost')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_total.groupby('Date')['인건비'].mean(), label='Total', linestyle='-', linewidth=2, color='#f15628')
    ax.plot(df_normal.groupby('Date')['인건비'].mean(), label='Normal', linestyle='-', linewidth=2, color='#1ca392')
    ax.plot(df_delay.groupby('Date')['인건비'].mean(), label='Delay', linestyle='-', linewidth=2, color='#ffc81b')
    ax.set_title(f'Labor Cost for {ship_type} ({start_date} ~ {end_date})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Labor cost')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_total.groupby('Date')['유류비(\)'].mean(), label='Total', linestyle='-', linewidth=2, color='#f15628')
    ax.plot(df_normal.groupby('Date')['유류비(\)'].mean(), label='Normal', linestyle='-', linewidth=2, color='#1ca392')
    ax.plot(df_delay.groupby('Date')['유류비(\)'].mean(), label='Delay', linestyle='-', linewidth=2, color='#ffc81b')
    ax.set_title(f'Fuel Cost for {ship_type} ({start_date} ~ {end_date})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Fuel cost')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

   # 총 경비 예측
    st.write("### 04 총 경비 예측")
    st.write("선종의 총 경비를 예측합니다. 총 경비를 예측하기 위해 각각의 항목들을 예측합니다.")

    # 인건비 예측
    st.write("#### 1. 인건비 예측")

    labor_cost_df = df_filtered.groupby('Date')['인건비'].mean().reset_index()
    labor_cost_df['Date'] = pd.to_datetime(labor_cost_df['Date'])
    labor_cost_df['Date_ordinal'] = labor_cost_df['Date'].map(lambda x: x.toordinal())

    X_labor = labor_cost_df[['Date_ordinal']]
    y_labor = labor_cost_df['인건비']
    labor_model = LinearRegression()
    labor_model.fit(X_labor, y_labor)

    last_actual_date = labor_cost_df['Date'].max()
    last_actual_date_num = mdates.date2num(last_actual_date)

    future_dates_labor = pd.date_range(start=last_actual_date + pd.DateOffset(months=1), end='2025-12-01', freq='MS')
    future_labor_df = pd.DataFrame({'Date': future_dates_labor})
    future_labor_df['Date_ordinal'] = future_labor_df['Date'].map(lambda x: x.toordinal())
    future_labor_df['인건비'] = labor_model.predict(future_labor_df[['Date_ordinal']])

    combined_labor_cost_df = pd.concat([
        labor_cost_df[['Date', '인건비']],
        future_labor_df[['Date', '인건비']]
    ], ignore_index=True)
    combined_labor_cost_df.sort_values('Date', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    dates = mdates.date2num(combined_labor_cost_df['Date'])
    costs = combined_labor_cost_df['인건비'].values
    points = np.array([dates, costs]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = ['blue' if date <= last_actual_date_num else 'orange' for date in dates[:-1]]

    lc = LineCollection(segments, colors=colors, linewidths=2)
	
    ax.add_collection(lc)
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(costs.min() - np.ptp(costs) * 0.1, costs.max() + np.ptp(costs) * 0.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
	
    plt.xticks(rotation=45)

    actual_patch = plt.Line2D([0], [0], color='blue', label='Labor cost')
    predicted_patch = plt.Line2D([0], [0], color='orange', label='Prediction cost')
    ax.legend(handles=[actual_patch, predicted_patch])

    ax.set_xlabel('Date')
    ax.set_ylabel('인건비')
    ax.set_title(f'Changes in labor cost over time ({ship_type})')

    plt.tight_layout()

    st.pyplot(fig)

    future_labor_df['Date_str'] = future_labor_df['Date'].dt.strftime('%Y-%m')
    predicted_labor_df = future_labor_df[future_labor_df['Date'].dt.year == 2025][['Date_str', '인건비']]
    predicted_labor_df.rename(columns={'Date_str': 'Date', '인건비': '예측 인건비'}, inplace=True)

    with st.expander("인건비 예측 결과 테이블 보기"):
        st.write("##### 인건비 예측 결과 (2025년 01월 ~ 2025년 12월)")
        st.table(predicted_labor_df.reset_index(drop=True))


    # 기타 비용 예측
    st.write("#### 2. 기타 비용 예측")

    other_cost_df = df_filtered.groupby('Date')['기타비용'].mean().reset_index()
    other_cost_df['Date'] = pd.to_datetime(other_cost_df['Date'])
    other_cost_df['Date_ordinal'] = other_cost_df['Date'].map(lambda x: x.toordinal())

    X_other = other_cost_df[['Date_ordinal']]
    y_other = other_cost_df['기타비용']
    other_model = LinearRegression()
    other_model.fit(X_other, y_other)

    last_actual_date = other_cost_df['Date'].max()
    last_actual_date_num = mdates.date2num(last_actual_date)

    future_dates_other = pd.date_range(start=last_actual_date + pd.DateOffset(months=1), end='2025-12-01', freq='MS')
    future_other_df = pd.DataFrame({'Date': future_dates_other})
    future_other_df['Date_ordinal'] = future_other_df['Date'].map(lambda x: x.toordinal())
    future_other_df['기타비용'] = other_model.predict(future_other_df[['Date_ordinal']])

    combined_other_cost_df = pd.concat([
        other_cost_df[['Date', '기타비용']],
        future_other_df[['Date', '기타비용']]
    ], ignore_index=True)
    combined_other_cost_df.sort_values('Date', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    dates = mdates.date2num(combined_other_cost_df['Date'])
    costs = combined_other_cost_df['기타비용'].values

    points = np.array([dates, costs]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = ['blue' if date <= last_actual_date_num else 'orange' for date in dates[:-1]]

    lc = LineCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)

    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(costs.min() - np.ptp(costs) * 0.1, costs.max() + np.ptp(costs) * 0.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    actual_patch = plt.Line2D([0], [0], color='blue', label='Other cost')
    predicted_patch = plt.Line2D([0], [0], color='orange', label='Prediction cost')
    ax.legend(handles=[actual_patch, predicted_patch])

    ax.set_xlabel('Date')
    ax.set_ylabel('기타비용')
    ax.set_title(f'Changes in other costs over time ({ship_type})')

    plt.tight_layout()
    st.pyplot(fig)

    future_other_df['Date_str'] = future_other_df['Date'].dt.strftime('%Y-%m')
    predicted_other_df = future_other_df[future_other_df['Date'].dt.year == 2025][['Date_str', '기타비용']]
    predicted_other_df.rename(columns={'Date_str': 'Date', '기타비용': '예측 기타비용'}, inplace=True)

    with st.expander("기타 비용 예측 결과 테이블 보기"):
        st.write("##### 기타 비용 예측 결과 (2025년 01월 ~ 2025년 12월)")
        st.table(predicted_other_df.reset_index(drop=True))


    # 유류비 예측
    st.write("#### 3. 유류비 예측")

    fuel_price_df = pd.read_excel('연도별 유류비.xlsx')
    fuel_price_df['Month'] = fuel_price_df['Month'].astype(str)
    fuel_price_df['Date'] = pd.to_datetime(fuel_price_df['Year'].astype(str) + fuel_price_df['Month'], format='%Y%B')
    fuel_price_df['Average_Fuel_Price'] = (fuel_price_df['HFO 단가(100L)'] + fuel_price_df['MFO 단가(100L)']) / 2
    fuel_price_df = fuel_price_df[['Date', 'Average_Fuel_Price']]
    fuel_price_df['Date'] = pd.to_datetime(fuel_price_df['Date'])
    fuel_price_df['Date_ordinal'] = fuel_price_df['Date'].map(lambda x: x.toordinal())

    X_price = fuel_price_df[['Date_ordinal']]
    y_price = fuel_price_df['Average_Fuel_Price']
    price_model = LinearRegression()
    price_model.fit(X_price, y_price)

    last_actual_date = fuel_price_df['Date'].max()
    last_actual_date_num = mdates.date2num(last_actual_date)

    future_dates_fuel = pd.date_range(start=last_actual_date + pd.DateOffset(months=1), end='2025-12-01', freq='MS')
    future_df = pd.DataFrame({'Date': future_dates_fuel})
    future_df['Date_ordinal'] = future_df['Date'].map(lambda x: x.toordinal())
    future_df['Average_Fuel_Price'] = price_model.predict(future_df[['Date_ordinal']])

    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
    df_filtered = df_filtered.merge(fuel_price_df[['Date', 'Average_Fuel_Price']], on='Date', how='left')
    df_filtered['Fuel_Cost_per_Day'] = df_filtered['유류비(\)'] / df_filtered['시운전 일수']
    df_filtered = df_filtered.dropna(subset=['Average_Fuel_Price', 'Fuel_Cost_per_Day'])
    df_filtered['Date_ordinal'] = df_filtered['Date'].map(lambda x: x.toordinal())
	
    X_fuel = df_filtered[['Average_Fuel_Price', 'Date_ordinal']]
    y_fuel = df_filtered['Fuel_Cost_per_Day']

    fuel_cost_model = LinearRegression()
    fuel_cost_model.fit(X_fuel, y_fuel)

    future_df['Fuel_Cost_per_Day_Predicted'] = fuel_cost_model.predict(future_df[['Average_Fuel_Price', 'Date_ordinal']])

    average_days = df_filtered['시운전 일수'].mean()
    future_df['유류비'] = future_df['Fuel_Cost_per_Day_Predicted'] * average_days
    fuel_cost_df = df_filtered.groupby('Date')['유류비(\)'].mean().reset_index()
    fuel_cost_df.rename(columns={'유류비(\)': '유류비'}, inplace=True)
    fuel_cost_df['Date'] = pd.to_datetime(fuel_cost_df['Date'])

    combined_fuel_cost_df = pd.concat([
        fuel_cost_df[['Date', '유류비']],
        future_df[['Date', '유류비']]
    ], ignore_index=True)
    combined_fuel_cost_df.sort_values('Date', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    dates = mdates.date2num(combined_fuel_cost_df['Date'])
    costs = combined_fuel_cost_df['유류비'].values
    points = np.array([dates, costs]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = ['blue' if date <= last_actual_date_num else 'orange' for date in dates[:-1]]

    lc = LineCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)

    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(costs.min() - np.ptp(costs) * 0.1, costs.max() + np.ptp(costs) * 0.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
	
    plt.xticks(rotation=45)
    actual_patch = plt.Line2D([0], [0], color='blue', label='Fuel cost')
    predicted_patch = plt.Line2D([0], [0], color='orange', label='Prediction cost')
	
    ax.legend(handles=[actual_patch, predicted_patch])
    ax.set_xlabel('Date')
    ax.set_ylabel('유류비')
    ax.set_title(f'Change in fuel cost over time ({ship_type})')

    plt.tight_layout()

    st.pyplot(fig)

    future_df['Date_str'] = future_df['Date'].dt.strftime('%Y-%m')
    predicted_fuel_df = future_df[future_df['Date'].dt.year == 2025][['Date_str', '유류비']]
    predicted_fuel_df.rename(columns={'Date_str': 'Date', '유류비': '예측 유류비'}, inplace=True)

    with st.expander("유류비 예측 결과 테이블 보기"):
        st.write("##### 유류비 예측 결과 (2025년 01월 ~ 2025년 12월)")
        st.table(predicted_fuel_df.reset_index(drop=True))


    # 총 경비 예측
    st.write("#### 4. 총 경비 예측")

    total_cost_df = df_filtered.groupby('Date')['총 경비'].mean().reset_index()
    total_cost_df['Date'] = pd.to_datetime(total_cost_df['Date'])
    # Date_ordinal은 더 이상 사용하지 않으므로 제거합니다.

    X_total = total_cost_df[['Date']]
    y_total = total_cost_df['총 경비']
    X_total['Date_ordinal'] = X_total['Date'].map(lambda x: x.toordinal())
    total_cost_model = LinearRegression()
    total_cost_model.fit(X_total[['Date_ordinal']], y_total)

    last_actual_date = total_cost_df['Date'].max()
    last_actual_date_num = mdates.date2num(last_actual_date)

    future_dates_extended = pd.date_range(start=last_actual_date + pd.DateOffset(months=1), end='2025-12-01', freq='MS')
    future_dates_df = pd.DataFrame({'Date': future_dates_extended})
    future_dates_df['Date_ordinal'] = future_dates_df['Date'].map(lambda x: x.toordinal())
    future_dates_df['총 경비'] = total_cost_model.predict(future_dates_df[['Date_ordinal']])

    combined_total_cost_df = pd.concat([
        total_cost_df[['Date', '총 경비']],
        future_dates_df[['Date', '총 경비']]
    ], ignore_index=True)
    combined_total_cost_df.sort_values('Date', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    dates = mdates.date2num(combined_total_cost_df['Date'])
    costs = combined_total_cost_df['총 경비'].values
    points = np.array([dates, costs]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = ['blue' if date <= last_actual_date_num else 'orange' for date in dates[:-1]]

    lc = LineCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)

    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(costs.min() - np.ptp(costs) * 0.1, costs.max() + np.ptp(costs) * 0.1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Cost')
    ax.set_title(f'Changes in total cost over time ({ship_type})')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    actual_patch = plt.Line2D([0], [0], color='blue', label='Total cost')
    predicted_patch = plt.Line2D([0], [0], color='orange', label='Prediction cost')
    ax.legend(handles=[actual_patch, predicted_patch])

    plt.tight_layout()
    st.pyplot(fig)

    future_dates_df['Date_str'] = future_dates_df['Date'].dt.strftime('%Y-%m')
    predicted_total_cost_df_2025 = future_dates_df[future_dates_df['Date'].dt.year == 2025][['Date_str', '총 경비']]
    predicted_total_cost_df_2025.rename(columns={'Date_str': 'Date', '총 경비': '예측 총 경비'}, inplace=True)

    with st.expander("총 경비 예측 결과 테이블 보기"):
        st.write("##### 총 경비 예측 결과 (2025년 01월 ~ 2025년 12월)")
        st.table(predicted_total_cost_df_2025.reset_index(drop=True)) 


#원인 분석
st.write("### 05. 원인 분석")
st.write("#### 2024년 평균 대비 2025년의 시운전 예측 비용을 원인 분석합니다.")

months = [f"{i}월" for i in range(1, 13)]
selected_month = st.selectbox("2025년 예측하고 싶은 월을 선택해주세요.", months, index=0)
selected_month_num = int(selected_month.replace('월', ''))
df_2024 = df_filtered[df_filtered['Date'].dt.year == 2024]

if df_2024.empty:
    st.write("선택한 선종의 2024년 데이터가 없습니다.")
else:
    average_2024_labor = df_2024['인건비'].mean()
    average_2024_other = df_2024['기타비용'].mean()
    average_2024_fuel = df_2024['유류비(\)'].mean()

    predicted_labor_df['Date'] = pd.to_datetime(predicted_labor_df['Date'], format='%Y-%m')
    labor_cost_2025_row = predicted_labor_df[(predicted_labor_df['Date'].dt.year == 2025) & (predicted_labor_df['Date'].dt.month == selected_month_num)]
    if labor_cost_2025_row.empty:
        st.write(f"선택한 월의 인건비 예측 데이터가 없습니다.")
        labor_cost_2025 = None
    else:
        labor_cost_2025 = labor_cost_2025_row['예측 인건비'].values[0]

    predicted_other_df['Date'] = pd.to_datetime(predicted_other_df['Date'], format='%Y-%m')
    other_cost_2025_row = predicted_other_df[(predicted_other_df['Date'].dt.year == 2025) & (predicted_other_df['Date'].dt.month == selected_month_num)]
    if other_cost_2025_row.empty:
        st.write(f"선택한 월의 기타비용 예측 데이터가 없습니다.")
        other_cost_2025 = None
    else:
        other_cost_2025 = other_cost_2025_row['예측 기타비용'].values[0]

    predicted_fuel_df['Date'] = pd.to_datetime(predicted_fuel_df['Date'], format='%Y-%m')
    fuel_cost_2025_row = predicted_fuel_df[(predicted_fuel_df['Date'].dt.year == 2025) & (predicted_fuel_df['Date'].dt.month == selected_month_num)]
    if fuel_cost_2025_row.empty:
        st.write(f"선택한 월의 유류비 예측 데이터가 없습니다.")
        fuel_cost_2025 = None
    else:
        fuel_cost_2025 = fuel_cost_2025_row['예측 유류비'].values[0]

    if None in [labor_cost_2025, other_cost_2025, fuel_cost_2025]:
        st.write("예측 비용 데이터가 부족하여 결과를 표시할 수 없습니다.")
    else:
        labor_increase_rate = ((labor_cost_2025 - average_2024_labor) / average_2024_labor * 100) if average_2024_labor != 0 else None
        other_increase_rate = ((other_cost_2025 - average_2024_other) / average_2024_other * 100) if average_2024_other != 0 else None
        fuel_increase_rate = ((fuel_cost_2025 - average_2024_fuel) / average_2024_fuel * 100) if average_2024_fuel != 0 else None

        def format_increase_rate(rate):
            return f"{rate:.2f}%" if rate is not None else "N/A"

        result_df = pd.DataFrame({
            '항목': ['인건비', '기타 비용', '유류비'],
            '2024년 평균 비용': [average_2024_labor, average_2024_other, average_2024_fuel],
            f'2025년 {selected_month} 예측 비용': [labor_cost_2025, other_cost_2025, fuel_cost_2025],
            '상승률': [format_increase_rate(labor_increase_rate), format_increase_rate(other_increase_rate), format_increase_rate(fuel_increase_rate)]
        })
        def format_currency(value):
            return f"{int(value):,}원" if not pd.isnull(value) else "N/A"

        result_df['2024년 평균 비용'] = result_df['2024년 평균 비용'].apply(format_currency)
        result_df[f'2025년 {selected_month} 예측 비용'] = result_df[f'2025년 {selected_month} 예측 비용'].apply(format_currency)

        st.table(result_df)

        increase_rates = [labor_increase_rate, other_increase_rate, fuel_increase_rate]
        valid_indices = [i for i, rate in enumerate(increase_rates) if rate is not None]

        if not valid_indices:
            st.write("상승률을 계산할 수 없습니다.")
        else:
            max_increase_index = max(valid_indices, key=lambda i: abs(increase_rates[i]))
            max_increase_rate = increase_rates[max_increase_index]
            max_increase_item = result_df.loc[max_increase_index, '항목']
            change_type = '상승' if max_increase_rate >= 0 else '감소'
            st.write(f"{ship_type} 선종의 **{max_increase_item}**이(가) 전체 총 경비의 **{abs(max_increase_rate):.2f}% {change_type}**으로 가장 큰 변동을 보인 항목입니다.")
	
	
    st.write("### 06. 항목별 비용 특성 분석 (Radar Chart)")
    st.write("유류비, 인건비, 총 경비, 기타비용의 정상 운영과 지연 운영 간의 비율을 비교하여 분석")
    with st.expander("Radar Chart 설명 보기"):
        st.write("#### 항목명 : Radar plot 에서 각 꼭지점")
        st.write("Fuel Cost: 시운전 선박이 사용한 유류비를 의미.")
        st.write("STN_C: Sea Trials Navigator Cost의 약자로 항해사비를 의미. 항해사비는 선장 비용, 타수 비용, 도선비, 임시항해 검사비, 자차 수정 비용이 포함된 금액.")
        st.write("SMMT_C: Ship Maintenance and Management Team의 약자로 노무비를 의미.")
        st.write("Other Cost: 시운전 선박의 기타 경비를 의미. 기타 비용은 용도품 침구 및 물품, 예선료, 통선비, 양식, 한식 이 포함된 금액.")
        st.write("Total Cost: 시운전 선박의 총경비를 의미.")
        st.write("#### 운영 결과")
        st.write("**Normal**: 해당 시운전이 계획된 기간에 맞게 진행이 되었음을 의미.")
        st.write("**Delay**: 해당 시운전이 계획된 기간(days)보다 지연이 되었음을 의미.")
        st.write("**Total**: 시운전 선박의 모든 데이터를 의미.")

    def calculate_cost_ratios(df):
        cost_columns = ['유류비(\)', '항해사비', '노무원비용', '총 경비', '기타비용']

        normal_df = df[df['지연 여부'] == '정상']
        delayed_df = df[df['지연 여부'] == '지연']

        total_avg = df[cost_columns].mean()

        normal_avg = normal_df[cost_columns].mean()
        delayed_avg = delayed_df[cost_columns].mean()

        ratios = {
            'Fuel Cost': {
                'Normal': (normal_avg['유류비(\)'] / total_avg['유류비(\)']) * 90,
                'Delay': (delayed_avg['유류비(\)'] / total_avg['유류비(\)']) * 110,
                'Total': 100
            },
	        'SMMT_C': {
                'Normal': (normal_avg['노무원비용'] / total_avg['노무원비용']) * 90,
                'Delay': (delayed_avg['노무원비용'] / total_avg['노무원비용']) * 110,
                'Total': 100
            },
            'Total Cost': {
                'Normal': (normal_avg['총 경비'] / total_avg['총 경비']) * 90,
                'Delay': (delayed_avg['총 경비'] / total_avg['총 경비']) * 110,
                'Total': 100
            },
            'Other Cost': {
                'Normal': (normal_avg['기타비용'] / total_avg['기타비용']) * 90,
                'Delay': (delayed_avg['기타비용'] / total_avg['기타비용']) * 110,
                'Total': 100
            },
            'STN_C': {
                'Normal': (normal_avg['항해사비'] / total_avg['항해사비']) * 90,
                'Delay': (delayed_avg['항해사비'] / total_avg['항해사비']) * 110,
                'Total': 100
            }
        }

        return ratios

    def plot_radar_chart(ratios):
        categories = ['Fuel Cost', 'SMMT_C', 'Total Cost', 'Other Cost', 'STN_C']

        normal_values = [ratios[cat]['Normal'] for cat in categories]
        delayed_values = [ratios[cat]['Delay'] for cat in categories]
        total_values = [ratios[cat]['Total'] for cat in categories]

        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        normal_values += normal_values[:1]
        delayed_values += delayed_values[:1]
        total_values += total_values[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories, fontsize=7, horizontalalignment='center')
        ax.set_rlabel_position(0)
        plt.yticks([60, 80, 100, 125], ["60%", "80%", "100%", "125%"], color="grey", size=6)
        plt.ylim(60, 125)

        ax.plot(angles, total_values, linewidth=1, linestyle='solid', label='Total', color='#f15628')
        ax.fill(angles, total_values, alpha=0.1, color='#f15628')
        ax.plot(angles, normal_values, linewidth=1, linestyle='solid', label='Normal', color='#1ca392')
        ax.fill(angles, normal_values, alpha=0.1, color='#1ca392')
        ax.plot(angles, delayed_values, linewidth=1, linestyle='solid', label='Delay', color='#ffc81b')
        ax.fill(angles, delayed_values, alpha=0.1, color='#ffc81b')

        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=8)
        plt.title('Radar Chart by Cost')
        st.pyplot(fig)

    ratios = calculate_cost_ratios(df_filtered)

    plot_radar_chart(ratios)
