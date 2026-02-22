import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="System predykcji wystąpienia cukrzycy",
    layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 400px;
        max-width: 800px;
    }
    [data-testid="stFormSubmitButton"] > button {
        background-color: #ff4b4b;
        color: white;
        border: none;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #ff0000;
        color: white;
    }
    .stAlert {
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_data():
    try:
        model = joblib.load('diabetes_model.joblib')
        scaler = joblib.load('diabetes_scaler.joblib')
        model_columns = joblib.load('model_columns.joblib')
        return model, scaler, model_columns
    except FileNotFoundError:
        return None, None, None


model, scaler, model_columns = load_data()

if model is None:
    st.error("Nie znaleziono plików modelu.")
    st.stop()

numeric_features = ['_BMI5', 'MENTHLTH', 'PHYSHLTH']

categorical_features = [
    '_RFHYPE6', 'TOLDHI3', 'SMOKE100', 'CVDSTRK3',
    '_MICHD', '_TOTINDA', '_HLTHPL1', 'MEDCOST1', 'GENHLTH',
    'DIFFWALK', 'SEXVAR', '_AGEG5YR', 'EDUCA', 'INCOME3'
]

age_options = {
    1.0: "18-24 lata", 2.0: "25-29 lat", 3.0: "30-34 lata", 4.0: "35-39 lat",
    5.0: "40-44 lata", 6.0: "45-49 lat", 7.0: "50-54 lata", 8.0: "55-59 lat",
    9.0: "60-64 lata", 10.0: "65-69 lat", 11.0: "70-74 lata", 12.0: "75-79 lat",
    13.0: "80 lat lub więcej",
}

educa_options = {
    1.0: "Brak wykształcenia", 2.0: "Podstawowe",
    3.0: "Gimnazjalne / Zasadnicze zawodowe", 4.0: "Średnie (Liceum / Technikum)",
    5.0: "Policealne / Wyższe niepełne", 6.0: "Wyższe (Licencjat / Magister)",
}

income_options = {
    1.0: "Poniżej 3 500 zł",
    2.0: "3 500 - 5 000 zł",
    3.0: "5 000 - 6 500 zł",
    4.0: "6 500 - 8 500 zł",
    5.0: "8 500 - 11 500 zł",
    6.0: "11 500 - 16 500 zł",
    7.0: "16 500 - 25 000 zł",
    9.0: "33 000 - 50 000 zł",
    10.0: "50 000 - 65 000 zł",
    11.0: "Powyżej 65 000 zł",
}

genhlth_options = {
    1.0: "Doskonałe", 2.0: "Bardzo dobre", 3.0: "Dobre", 4.0: "Średnie", 5.0: "Złe"
}

st.sidebar.title("Formularz Diagnostyczny")
st.sidebar.write("Uzupełnij dane:")

input_data = {}

with st.sidebar.form("diabetes_form"):
    with st.expander("Dane Demograficzne", expanded=True):
        input_data['SEXVAR'] = st.radio("Płeć", [1.0, 2.0],
                                        format_func=lambda x: "Mężczyzna" if x == 1.0 else "Kobieta",
                                        index=0)
        input_data['_AGEG5YR'] = st.selectbox("Wiek", options=list(age_options.keys()),
                                              format_func=lambda x: age_options[x], index=0)
        input_data['EDUCA'] = st.selectbox("Wykształcenie", options=list(educa_options.keys()),
                                           format_func=lambda x: educa_options[x], index=0)
        input_data['INCOME3'] = st.selectbox("Miesięczny dochód", options=list(income_options.keys()),
                                             format_func=lambda x: income_options[x], index=0)

    with st.expander("Stan Zdrowia"):
        st.markdown("**Wskaźniki zdrowotne**")

        bmi_input = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1,
                                    help="Wskaźnik proporcji masy ciała do wzrostu. Wynik powyżej 25 oznacza nadwagę, a powyżej 30 otyłość. Oblicza się go za pomocą wzoru: kg/m².")
        input_data['_BMI5'] = bmi_input * 100

        with st.expander("Kalkulator BMI"):
            st.caption("Wpisz swój wzrost i wagę, aby poznać swoje BMI.")
            col_calc1, col_calc2 = st.columns(2)
            with col_calc1:
                calc_height = st.number_input("Wzrost (cm)", min_value=50, max_value=250, value=175)
            with col_calc2:
                calc_weight = st.number_input("Waga (kg)", min_value=20, max_value=300, value=75)

            if st.form_submit_button("Przelicz BMI", use_container_width=True):
                calculated_bmi = calc_weight / ((calc_height / 100) ** 2)
                st.info(f"Twoje BMI wynosi: **{calculated_bmi:.2f}**")

        input_data['GENHLTH'] = st.select_slider("Ogólna ocena zdrowia", options=[1.0, 2.0, 3.0, 4.0, 5.0], value=1.0,
                                                 format_func=lambda x: genhlth_options[x],
                                                 help="Subiektywnie oceń swoje zdrowie, biorąc pod uwagę Twoją kondycję fizyczną, poziom energii i samopoczucie.")
        input_data['PHYSHLTH'] = st.slider("Dni złego samopoczucia fizycznego w ostatnim miesiącu", 0, 30, 0)
        input_data['MENTHLTH'] = st.slider("Dni złego samopoczucia psychicznego w ostatnim miesiącu", 0, 30, 0)

        st.divider()
        st.markdown("**Problemy zdrowotne**")
        st.caption("Zaznacz pola, jeśli odpowiedź brzmi TAK:")
        input_data['_RFHYPE6'] = 2.0 if st.checkbox("Czy zdiagnozowanu u Ciebie nadciśnienie tętniczne?",
                                                    value=False) else 1.0
        input_data['TOLDHI3'] = 1.0 if st.checkbox("Czy zdiagnozowano u Ciebie wysoki poziom cholesterolu?",
                                                   value=False) else 2.0
        input_data['CVDSTRK3'] = 1.0 if st.checkbox("Czy Przebyłeś/aś udar mózgu?", value=False) else 2.0
        input_data['_MICHD'] = 1.0 if st.checkbox("Czy masz choroba wieńcową lub przebyłeś zawał serca?",
                                                  value=False) else 2.0
        input_data['DIFFWALK'] = 1.0 if st.checkbox("Czy masz trudności z chodzeniem?", value=False) else 2.0

    with st.expander("Styl Życia i Opieka Medyczna"):
        st.caption("Zaznacz pola, jeśli odpowiedź brzmi TAK:")
        input_data['SMOKE100'] = 2.0 if st.checkbox("Czy wypaliłeś/aś w swoim życiu co najmniej 100 papierosów?",
                                                    value=False) else 1.0
        input_data['_TOTINDA'] = 1.0 if st.checkbox(
            "Czy uprawiałeś jakąkolwiek aktywność fizyczną w ciągu ostatnich 30 dni?", value=False) else 2.0
        st.divider()
        input_data['_HLTHPL1'] = 2.0 if st.checkbox("Czy posiadasz ubezpieczenie zdrowotne?", value=False) else 1.0
        input_data['MEDCOST1'] = 1.0 if st.checkbox(
            "Czy w ciągu roku zrezygnowałeś/aś z wizyty lekarskiej z przyczyn finansowych?", value=False) else 2.0

    submit_button = st.form_submit_button("Oblicz Ryzyko", type="primary", use_container_width=True)

st.title("Kalkulator Ryzyka Wystąpienia Cukrzycy")
st.markdown("Szacowanie ryzyka zachorowania na cukrzycę przy użyciu algorytmu uczenia maszynowego.")
st.divider()

if submit_button:
    df_input = pd.DataFrame([input_data])
    df_encoded = pd.get_dummies(df_input, columns=categorical_features, drop_first=False)
    df_aligned = df_encoded.reindex(columns=model_columns, fill_value=0)

    try:
        df_aligned[numeric_features] = scaler.transform(df_aligned[numeric_features])
    except Exception as e:
        st.error(f"Błąd skalowania danych: {e}")
        st.stop()

    try:
        prediction_prob = model.predict_proba(df_aligned)[0][1]
        prob_percent = prediction_prob * 100
    except Exception as e:
        st.error(f"Błąd predykcji modelu: {e}")
        st.stop()

    feature_map = {
        "_BMI5": "BMI",
        "SEXVAR_1.0": "Płeć",
        "SEXVAR_2.0": "Płeć",
        "_RFHYPE6_2.0": "Nadciśnienie",
        "_RFHYPE6_1.0": "Brak nadciśnienia",
        "TOLDHI3_1.0": "Wysoki cholesterol",
        "TOLDHI3_2.0": "Prawidłowy cholesterol",
        "_CHOLCH3_1.0": "Zbadany cholesterol",
        "_CHOLCH3_2.0": "Brak badań cholesterolu",
        "_MICHD_1.0": "Choroba serca lub Zawał",
        "_MICHD_2.0": "Zdrowe serce",
        "CVDSTRK3_1.0": "Przebyty udar",
        "CVDSTRK3_2.0": "Brak udarów",
        "DIFFWALK_1.0": "Trudności z chodzeniem",
        "DIFFWALK_2.0": "Sprawność ruchowa",
        "SMOKE100_2.0": "Palenie papierosów",
        "SMOKE100_1.0": "Osoba niepaląca",
        "_RFDRHV8_2.0": "Nadużywanie alkoholu",
        "_RFDRHV8_1.0": "Brak problemów z alkoholem",
        "_TOTINDA_1.0": "Uprawianie aktywności fizycznej",
        "_TOTINDA_2.0": "Brak aktywności fizycznej",
        "_HLTHPL1_2.0": "Ubezpieczony",
        "_HLTHPL1_1.0": "Brak ubezpieczenia",
        "MEDCOST1_1.0": "Rezygnacja z leczenia przez koszty",
        "MEDCOST1_2.0": "Stała opieka lekarska",
        "GENHLTH_1.0": "Ogólny stan zdrowia",
        "GENHLTH_2.0": "Ogólny stan zdrowia",
        "GENHLTH_3.0": "Ogólny stan zdrowia",
        "GENHLTH_4.0": "Ogólny stan zdrowia",
        "GENHLTH_5.0": "Ogólny stan zdrowia",
    }


    def get_clean_name(feat_name):
        if feat_name in feature_map:
            return feature_map[feat_name]

        if "_AGEG5YR_" in feat_name:
            return "Wiek"

        if "_BMI5" in feat_name:
            return "BMI"

        if "EDUCA_" in feat_name: return "Wykształcenie"
        if "INCOME3_" in feat_name: return "Dochód"

        return feat_name


    feature_names = model_columns
    coefficients = model.coef_[0]
    input_values = df_aligned.values[0]

    contributions = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': coefficients * input_values
    })

    hidden_patterns = ['MENTHLTH', 'PHYSHLTH', 'MEDCOST', 'INCOME3']
    mask = contributions['Feature'].apply(lambda x: any(p in x for p in hidden_patterns))
    contributions = contributions[~mask].copy()

    contributions = contributions[contributions['Contribution'].abs() > 0.0001].copy()
    contributions['CleanName'] = contributions['Feature'].apply(get_clean_name)
    plot_data = contributions.sort_values(by='Contribution', ascending=True)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Ocena Ryzyka")
        if prob_percent < 30:
            st.success(f"Niskie: {prob_percent:.2f}%")
        elif prob_percent < 50:
            st.warning(f"Umiarkowane: {prob_percent:.2f}%")
        else:
            st.error(f"Wysokie: {prob_percent:.2f}%")

        st.divider()
        st.subheader("Wnioski i zalecenia")
        st.markdown("Sugestie oparte na wprowadzonych przez Ciebie danych")


        def recommendation_card(title, icon, text, color_border="#eee"):
            st.markdown(f"""
            <div style="
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid {color_border};
                background-color: #262730; 
                margin-bottom: 15px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                <h4 style="margin:0; font-size: 18px;">{icon} {title}</h4>
                <p style="margin:5px 0 0 0; font-size: 14px; opacity: 0.9;">{text}</p>
            </div>
            """, unsafe_allow_html=True)


        recs_found = False

        if bmi_input >= 30:
            recommendation_card("Redukcja wagi", '',
                                "Twoje BMI wskazuje na otyłość. Nawet niewielka redukcja masy ciała drastycznie zmniejsza ryzyko cukrzycy.",
                                "#ff4b4b")
            recs_found = True
        elif bmi_input >= 25:
            recommendation_card("Kontrola wagi", '',
                                "Twoja waga jest nieco wyższa niż zalecana. Warto przyjrzeć się diecie oraz wyeliminować puste kalorie.",
                                "#ffa726")
            recs_found = True

        if input_data['SMOKE100'] == 2.0:
            recommendation_card("Rzucenie palenia", '',
                                "Nikotyna zaburza gospodarkę cukrową organizmu i zwiększa insulinooporność.",
                                "#ff4b4b")
            recs_found = True

        if input_data['_TOTINDA'] == 2.0:
            recommendation_card("Aktywność fizyczna", '',
                                "Zwykły, codzienny spacer sprawia, że mięśnie spalają glukozę i naturalnie odciążają Twój organizm.",
                                "#29b5e8")
            recs_found = True

        if input_data['_RFHYPE6'] == 2.0:
            recommendation_card("Ciśnienie krwi", '',
                                "Nadciśnienie często idzie w parze z cukrzycą. Regularne pomiary i leczenie są kluczowe.",
                                "#ffa726")
            recs_found = True

        if input_data['TOLDHI3'] == 1.0:
            recommendation_card("Niższy cholesterol", '',
                                "Jedzenie dużej ilości tłuszczów zwierzęcych zwiększa cholesterol. Dla zdrowia lepiej używać tłuszczów roślinnych.", "#ffa726")
            recs_found = True

        if not recs_found:
            recommendation_card("Dobre nawyki", '',
                                "Twój styl życia jest korzystny dla zdrowia. Warto to kontynuować, aby utrzymać niskie ryzyko.",
                                "#00c853")


    with col2:
        st.subheader("Szczegółowa analiza czynników")

        colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in plot_data['Contribution']]

        if not plot_data.empty:
            max_val = plot_data['Contribution'].abs().max()
            axis_limit = max_val * 1.25
        else:
            axis_limit = 1.0

        fig_factors = go.Figure()

        fig_factors.add_trace(go.Bar(
            x=plot_data['Contribution'],
            y=plot_data['CleanName'],
            orientation='h',
            marker_color=colors,
            marker_line_color=colors,
            marker_line_width=1.5,
            opacity=0.9,
            text=plot_data['Contribution'].apply(lambda x: f"{x:+.2f}"),
            textposition='outside',
            textfont=dict(size=12, color='white'),
            hovertemplate="<b>%{y}</b><br>Wpływ: %{x:.4f}<extra></extra>"
        ))

        fig_factors.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=True,
                zerolinecolor='white',
                zerolinewidth=2,
                showticklabels=True,
                range=[-axis_limit, axis_limit]
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(size=13, color='white'),
                automargin=True
            ),
            showlegend=False,
            font=dict(family="Source Sans Pro, sans-serif")
        )

        st.plotly_chart(fig_factors, use_container_width=True)

        with st.expander("Jak interpretować wykres?"):
            st.markdown("""
            Model analizuje wpływ każdej Twojej odpowiedzi na ostateczny wynik:
            * <span style='color:#2ecc71'><b> Zielone słupki</b></span> – oznaczają cechy, które **obniżają ryzyko**.
            * <span style='color:#e74c3c'><b> Czerwone słupki</b></span> – oznaczają cechy, które **zwiększają ryzyko**.
            """, unsafe_allow_html=True)

else:
    st.info('Wypełnij formularz w menu bocznym i kliknij przycisk "Oblicz Ryzyko", aby zobaczyć wynik.')