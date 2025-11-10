import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random

st.set_page_config(page_title="IoT Analiza Grijanja", layout="wide")

st.sidebar.title("Meni")
opcija = st.sidebar.radio("Odaberi prikaz scenarija optimizacije:", [
    "Prvi slučaj: IT WIFI v2 + kotlovi drvo/pelet",
    "Drugi slučaj: IT IRWIFI v2 termostat + IC paneli",
    "Treći slučaj: IT-200 termostat + električno grijanje",
    "Preporuke za dalji razvoj"
])

# ----------------- PRVI SLUČAJ -----------------
if opcija == "Prvi slučaj: IT WIFI v2 + kotlovi drvo/pelet":
    st.title("Optimizacija rada sistema grijanja – IT WIFI v2 termostat")
    st.image("semakuce1.png", caption="Šema analiziranog objekta")

    st.subheader("Potrošnja prije implementacije termostata")
    df = pd.read_excel("tabela6.xlsx")
    st.table(df)

    st.subheader("Potrošnja poslije implementacije termostata")
    df = pd.read_excel("tabela7.xlsx")
    st.table(df)

    st.subheader("Dijagram potrošnje prije i poslije implementacije termostata")
    st.image("prvislucaj.png", caption="Poređenje potrošnje prije i poslije implementacije IT WIFI v2 termostata")

# ----------------- DRUGI SLUČAJ -----------------
elif opcija == "Drugi slučaj: IT IRWIFI v2 termostat + IC paneli":
    st.title("Implementacija IoT uređaja sa sistemom električnog grijanja – IT IRWIFI v2")

    st.subheader("Rezultati poređenja tradicionalnog grijanja i IoT sistema sa toplotnim panelima")
    df = pd.read_excel("tabela8.xlsx")
    st.table(df)

    st.subheader("Dijagram poređenja tradicionalnog grijanja i IoT sistema sa toplotnim panelima")
    st.image("drugislucaj.png", caption="Dijagram poređenja tradicionalnog grijanja i IoT sistema sa toplotnim panelima")

# ----------------- TREĆI SLUČAJ -----------------
elif opcija == "Treći slučaj: IT-200 termostat + električno grijanje":
    st.title("Implementacija IoT uređaja – IT-200 termostat")
    st.image("semakuce2.png", caption="Šema kuće sa IT-200 termostatom")

    st.subheader("Dnevna potrošnja")
    st.image("potrosnjaapp1.png", caption="Dnevna potrošnja (kWh)")

    st.subheader("Mjesečna potrošnja")
    st.image("potrosnjaapp2.png", caption="Mjesečna potrošnja (kWh)")

    st.subheader("Energetska potrošnja i vremenski uslovi korištenjem IT - 200 termostata i električnog grijanja")
    df = pd.read_excel("tabela9.xlsx")
    st.table(df)


# ----------------- PREPORUKE -----------------
elif opcija == "Preporuke za dalji razvoj":
    st.title("Preporuke za dalji razvoj sistema")

    st.markdown("### 1. Povlačenje vremenske prognoze (7 dana)")
    API_KEY = "9447fb31f57d19d0888a15bc7f75ba81" 
    CITY = "Sarajevo"
    URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY},BA&appid={API_KEY}&units=metric"

    try:
        response = requests.get(URL)
        data = response.json()

        if response.status_code == 200:
            daily_data = data["list"][::8]
            datumi, temp_min, temp_max, opis = [], [], [], []

            for dan in daily_data:
                datumi.append(datetime.datetime.fromtimestamp(dan["dt"]).strftime("%d.%m"))
                temp_min.append(dan["main"]["temp_min"])
                temp_max.append(dan["main"]["temp_max"])
                opis.append(dan["weather"][0]["description"])

            prognoza_df = pd.DataFrame({
                "Datum": datumi,
                "Min temperatura (°C)": temp_min,
                "Max temperatura (°C)": temp_max,
                "Opis": opis
            })

            st.table(prognoza_df)

            fig, ax = plt.subplots()
            ax.plot(datumi, temp_min, marker='o', label="Min temperatura", color="blue")
            ax.plot(datumi, temp_max, marker='o', label="Max temperatura", color="orange")
            ax.set_title(f"Sedmični trend temperatura za {CITY}")
            ax.set_ylabel("Temperatura (°C)")
            ax.legend()
            st.pyplot(fig)

            # Preporučeni periodi grijanja
            st.subheader("Preporučeni periodi grijanja")
            preporuke = ["Povećati grijanje" if t < 5 else "Normalan režim" for t in temp_min]

            preporuke_df = pd.DataFrame({
                "Datum": datumi,
                "Min temperatura (°C)": temp_min,
                "Preporuka": preporuke
            })

            st.table(preporuke_df)

        else:
            st.error("Ne mogu preuzeti podatke. Provjeri API ključ ili naziv grada.")

    except Exception as e:
        st.error(f"Greška pri preuzimanju vremenske prognoze: {e}")

    st.markdown("### 2. LSTM prijedlog za optimizaciju")
    from ltsmdioapp import run_lstm_demo, plot_results
    res = run_lstm_demo(days=240, seq_len=14, forecast_horizon=7)
    st.subheader("LSTM: Predikcija potrošnje (demo na sintetičkim podacima)")
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_results(res, ax=ax) 
    st.pyplot(fig)
    
        # --- Numerička evaluacija LSTM modela ---
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Pretvaranje podataka iz rezultata
    y_true = res['y_true'][1]
    y_pred = res['y_pred'][1]

    # Izračun metrika
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    avg_true = np.mean(y_true)
    avg_pred = np.mean(y_pred)

    # Prikaz u Streamlitu
    st.subheader("📊 Numerička evaluacija LSTM modela")
    eval_data = pd.DataFrame({
        "Metrika": [
            "MAE (Mean Absolute Error)",
            "RMSE (Root Mean Squared Error)",
            "MAPE (%)",
            "Prosječna stvarna potrošnja (kWh/dan)",
            "Prosječna predviđena potrošnja (kWh/dan)"
        ],
        "Vrijednost": [
            f"{mae:.3f}",
            f"{rmse:.3f}",
            f"{mape:.2f}",
            f"{avg_true:.2f}",
            f"{avg_pred:.2f}"
        ]
    })

    st.table(eval_data)


