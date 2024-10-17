import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import re

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
            st.markdown("### 정상, Delay, 총합 카운트 (단위: 건)")
            st.table(count_data)

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

    st.write("### 04. 항목별 비용 특성 분석 (Radar Chart)")
    st.write("유류비, 인건비, 총 경비, 기타비용의 정상 운영과 지연 운영 간의 비율을 비교하여 분석")
    with st.expander("Radar Chart 설명 보기"):
        st.write("#### 항목명 : Radar plot 에서 각 꼭지점")
        st.write("Fuel Cost: 시운전 선박이 사용한 기름값을 나타냅니다.")
        st.write("STN_C: Sea Trials Navigator Cost의 약자로 항해사비가 됩니다. 항해사비는 선장 비용, 타수 비용, 도선비, 임시항해 검사비, 자차 수정 비용이 포함된 금액입니다.")
        st.write("SMMT_C: Ship Maintenance and Management Team의 약자로 노무비가 됩니다.")
        st.write("Other Cost: 시운전 선박의 기타 경비를 나타냅니다. 기타 비용은 용도품 침구 및 물품, 예선료, 통선비, 양식, 한식이 포함된 금액입니다.")
        st.write("Total Cost: 시운전 선박의 총경비를 나타냅니다.")
        st.write("#### 운영 결과")
        st.write("**Normal**: 시운전 선박의 정상 운영을 의미합니다.")
        st.write("**Delay**: 시운전 선박의 지연 운영을 의미합니다.")
        st.write("**Total**: 시운전 선박의 모든 데이터를 의미합니다.")

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
