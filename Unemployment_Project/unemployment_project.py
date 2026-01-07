

import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA


@st.cache_data
def load_data():
    df_state = pd.read_csv("data/Unemployment in India.csv")
    df_national = pd.read_csv("data/Unemployment_Rate_upto_11_2020.csv")

    df_state.columns = df_state.columns.str.strip()
    df_national.columns = df_national.columns.str.strip()

  
    df_state['Date'] = pd.to_datetime(df_state['Date'])
    df_national['Date'] = pd.to_datetime(df_national['Date'])

    return df_state.dropna(), df_national.dropna()

df_state, df_national = load_data()

st.title("📊 Real-Time Unemployment Analytics Dashboard")
st.write("Using State-wise & National Unemployment Data (India)")


st.subheader("📁 Dataset Preview")
st.write("State-wise Data")
st.dataframe(df_state.head())

st.write("National Level Data")
st.dataframe(df_national.head())


st.subheader("📈 National Unemployment Trend")

national_trend = df_national.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
st.line_chart(national_trend)


st.subheader("🦠 COVID-19 Impact Analysis")

pre_covid = df_national[df_national['Date'] < '2020-03-01']
post_covid = df_national[df_national['Date'] >= '2020-03-01']

pre_avg = pre_covid['Estimated Unemployment Rate (%)'].mean()
post_avg = post_covid['Estimated Unemployment Rate (%)'].mean()

col1, col2 = st.columns(2)
col1.metric("Pre-COVID Avg", f"{pre_avg:.2f}%")
col2.metric("Post-COVID Avg", f"{post_avg:.2f}%", delta=f"{post_avg-pre_avg:.2f}%")


st.subheader("🏛 State-wise Unemployment Analysis")

states = st.multiselect(
    "Select States",
    df_state['Region'].unique(),
    default=df_state['Region'].unique()[:5]
)

state_data = df_state[df_state['Region'].isin(states)]
state_trend = state_data.pivot_table(
    values='Estimated Unemployment Rate (%)',
    index='Date',
    columns='Region'
)

st.line_chart(state_trend)


st.subheader("🏙 Urban vs 🌾 Rural Comparison")

area_avg = df_state.groupby('Area')['Estimated Unemployment Rate (%)'].mean()
st.bar_chart(area_avg)


st.subheader("📅 Seasonal Trend")

df_state['Month'] = df_state['Date'].dt.month
monthly_avg = df_state.groupby('Month')['Estimated Unemployment Rate (%)'].mean()
st.line_chart(monthly_avg)


st.subheader("🔮 Unemployment Forecast (Next 12 Months)")

model = ARIMA(national_trend, order=(2,1,2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)

st.line_chart(forecast)

st.subheader("🧠 Policy Recommendation")

avg_rate = national_trend.mean()

if avg_rate > 10:
    st.error("🚨 Emergency employment programs required")
elif avg_rate > 6:
    st.warning("⚠ Skill development & MSME support needed")
else:
    st.success("✅ Stable economy – focus on innovation")

