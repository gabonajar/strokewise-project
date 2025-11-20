import streamlit as st
import streamlit.components.v1 as components
import base64
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Strokewise™", layout="wide")

# ---------- Multilingual text store ----------
LANGS = {
    'en': {
        'company': 'Strokewise™',
        'tagline': 'Smart, early prediction for stroke prevention.',
        'intro1': 'Enter your details to see your risk — because every second counts.',
        'intro2': 'Our next-generation platform makes prevention proactive and personal.',
        'form_title': 'Your Details',
        'age': 'Age',
        'gender': 'Gender',
        'male': 'Male',
        'female': 'Female',
        'other': 'Other',
        'hypertension': 'Do you have hypertension?',
        'yes': 'Yes',
        'no': 'No',
        'heart_disease': 'Any heart disease diagnosis?',
        'avg_glucose': 'Average glucose (mg/dL)',
        'bmi': 'BMI (kg/m²)',
        'smoker': 'Are you a smoker?',
        'smoke_never': 'Never',
        'smoke_former': 'Formerly',
        'smoke_current': 'Currently',
        'submit': 'Estimate My Stroke Risk',
        'result_title': 'Estimated Stroke Risk',
        'probability': 'Probability',
        'graph_title': 'Your Health Statistics',
        'graph_sub': 'Compared to population averages',
        'advice_low': 'Your estimated stroke risk is LOW. Keep up the healthy habits.',
        'advice_medium': 'Your risk is ELEVATED. Exercise, good nutrition and regular health checks are key.',
        'advice_high': 'Your risk is HIGH. Please seek medical advice for targeted prevention.',
        'advice_smoke': 'Quitting smoking reduces stroke risk dramatically.',
        'advice_hypertension': 'Managing blood pressure is crucial for prevention.',
        'advice_glucose': 'Controlling glucose levels helps lower risk.',
        'advice_bmi': 'Keeping BMI in a healthy range is protective.',
        'warning': 'This is an early warning estimate. Please consult your doctor for medical advice.',
    },
    'es': {
        'company': 'Strokewise™',
        'tagline': 'Predicción inteligente y temprana para la prevención de accidentes cerebrovasculares.',
        'intro1': 'Ingrese sus datos para ver su riesgo — porque cada segundo cuenta.',
        'intro2': 'Nuestra plataforma de última generación hace que la prevención sea proactiva y personalizada.',
        'form_title': 'Sus datos',
        'age': 'Edad',
        'gender': 'Género',
        'male': 'Masculino',
        'female': 'Femenino',
        'other': 'Otro',
        'hypertension': '¿Tiene hipertensión?',
        'yes': 'Sí',
        'no': 'No',
        'heart_disease': '¿Algún diagnóstico de enfermedad cardíaca?',
        'avg_glucose': 'Glucosa promedio (mg/dL)',
        'bmi': 'IMC (kg/m²)',
        'smoker': '¿Fuma usted?',
        'smoke_never': 'Nunca',
        'smoke_former': 'Anteriormente',
        'smoke_current': 'Actualmente',
        'submit': 'Estimar mi riesgo de ACV',
        'result_title': 'Riesgo estimado de ACV',
        'probability': 'Probabilidad',
        'graph_title': 'Sus estadísticas de salud',
        'graph_sub': 'Comparadas con promedios poblacionales',
        'advice_low': 'Su riesgo estimado de ACV es BAJO. Continúe con los hábitos saludables.',
        'advice_medium': 'Su riesgo es ELEVADO. El ejercicio, una buena nutrición y revisiones periódicas son esenciales.',
        'advice_high': 'Su riesgo es ALTO. Consulte a su médico para prevención específica.',
        'advice_smoke': 'Dejar de fumar reduce drásticamente el riesgo de ACV.',
        'advice_hypertension': 'Controlar la presión arterial es fundamental para la prevención.',
        'advice_glucose': 'Controlar la glucosa ayuda a reducir el riesgo.',
        'advice_bmi': 'Mantener un IMC saludable ayuda a protegerse.',
        'warning': 'Esta es una estimación preliminar, consulte a su médico para orientación profesional.',
    }
}

# Initialize session state for language
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

# --------------------- Language selector at top ---------------------
col_lang1, col_lang2, col_lang3 = st.columns([1, 10, 1])
with col_lang1:
    if st.button("English", key="btn_en", use_container_width=True):
        st.session_state.lang = 'en'
        st.rerun()
with col_lang3:
    if st.button("Español", key="btn_es", use_container_width=True):
        st.session_state.lang = 'es'
        st.rerun()

T = LANGS[st.session_state.lang]

# --------------------- Background video ---------------------
try:
    with open("background.mp4", "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode()

    st.markdown(f"""
        <style>
        .stApp {{
            background: transparent;
        }}
        .bgvid {{
            position: fixed;
            top: 0;
            left: 0;
            min-width: 100vw;
            min-height: 100vh;
            object-fit: cover;
            z-index: -100;
            opacity: 0.18;
        }}
        .hero-card {{
            background: #fff;
            border-radius: 30px;
            box-shadow: 0 12px 60px rgba(26, 24, 83, 0.5);
            margin: 2.7rem auto 3.5rem auto;
            padding: 2.8rem 3.2rem 2.2rem 3.2rem;
            max-width: 900px;
            text-align: left;
            position: relative;
        }}
        .hero-card h1 {{
            font-family: Montserrat, sans-serif;
            font-weight: 900;
            font-size: 3.1rem;
            letter-spacing: 2px;
            margin-bottom: 0.55rem;
            color: #23204b;
        }}
        .hero-card h2 {{
            font-size: 1.31rem;
            font-family: Montserrat, sans-serif;
            font-weight: 700;
            margin-bottom: 0.9rem;
            color: #5647a3;
        }}
        .hero-card .desc {{
            color: #15131b;
            font-size: 1.09rem;
            font-weight: 500;
            margin-bottom: 0.24rem;
        }}
        .hero-card .subdesc {{
            color: #6e6892;
            font-size: 0.97rem;
            margin-bottom: 0.18rem;
        }}
        .main-section {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }}
        .gauge-headline {{
            font-weight: 900;
            font-size: 2.2rem;
            margin-bottom: 0.8rem;
            color: #49e572;
            letter-spacing: 1.8px;
            text-align: center;
        }}
        .advice-section {{
            background: #e7e6fa;
            border-radius: 16px;
            padding: 1.3rem 2.1rem;
            margin: 36px 0 0 0;
        }}
        .warning-section {{
            background: #fadce7;
            border-radius: 13px;
            padding: 1.1rem 1.9rem;
            margin: 22px 0;
            font-size: 1.12rem;
            color: #be1e40;
            font-weight: 900;
            box-shadow: 0 8px 28px rgba(190, 30, 64, 0.07);
            text-align: center;
        }}
        </style>
        <video class="bgvid" autoplay loop muted playsinline>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4" />
        </video>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Background video not found. Please ensure 'background.mp4' is in the same directory.")

# --------- HERO SECTION: white card with branding -----------
st.markdown(f"""
    <div class="hero-card">
      <h1>{T['company']}</h1>
      <h2>{T['tagline']}</h2>
      <div class="desc">{T['intro1']}</div>
      <div class="subdesc">{T['intro2']}</div>
    </div>
""", unsafe_allow_html=True)

# --------- Main Content: Form, Results, Advice -------------
st.markdown("<div class='main-section'>", unsafe_allow_html=True)

with st.form(key='stroke_form'):
    st.markdown(f"<h4 style='margin-top:0.2rem;color:#4a4079'>{T['form_title']}</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider(T['age'], min_value=18, max_value=100, value=50)
        gender = st.selectbox(T['gender'], [T['male'], T['female'], T['other']])
        hypertension = st.selectbox(T['hypertension'], [T['no'], T['yes']])
        heart_disease = st.selectbox(T['heart_disease'], [T['no'], T['yes']])

    with col2:
        avg_glucose = st.number_input(T['avg_glucose'], min_value=50.0, max_value=350.0, value=100.0, step=0.1)
        bmi = st.number_input(T['bmi'], min_value=10.0, max_value=55.0, value=22.0, step=0.1)
        smoker = st.selectbox(T['smoker'], [T['smoke_never'], T['smoke_former'], T['smoke_current']])

    submitted = st.form_submit_button(T['submit'])

# ------- Result section: animated circle, graphs, advice ------
if submitted:
    risk_score = int(np.clip(np.random.normal(11, 7), 2, 38))  # Replace with your ML model
    risk_level = ['Low', 'Medium', 'High'][min(int(risk_score / 13), 2)]

    # ----- Animated percentage circle using HTML component -----
    st.markdown(f"<div class='gauge-headline'>{T['result_title']}</div>", unsafe_allow_html=True)

    gauge_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 280px;
                background: transparent;
            }}
            .gauge-container {{
                position: relative;
                width: 220px;
                height: 220px;
            }}
            #gauge-label {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 3.2rem;
                color: #49e572;
                font-weight: 900;
                font-family: Arial, sans-serif;
            }}
        </style>
    </head>
    <body>
        <div class="gauge-container">
            <canvas id="gauge-canvas" width="220" height="220"></canvas>
            <div id="gauge-label">0%</div>
        </div>
        <script>
            const gauge = document.getElementById('gauge-canvas');
            const ctx = gauge.getContext('2d');

            function drawGauge(percent) {{
                ctx.clearRect(0, 0, 220, 220);

                // Draw filled arc (green)
                ctx.beginPath();
                ctx.arc(110, 110, 85, Math.PI * 0.75, Math.PI * 0.75 + 2 * Math.PI * percent / 100, false);
                ctx.lineWidth = 23;
                ctx.strokeStyle = '#49e572';
                ctx.stroke();

                // Draw empty arc (gray)
                ctx.beginPath();
                ctx.arc(110, 110, 85, Math.PI * 0.75 + 2 * Math.PI * percent / 100, Math.PI * 2.75, false);
                ctx.lineWidth = 23;
                ctx.strokeStyle = '#efefef';
                ctx.stroke();
            }}

            let cur = 0;
            const target = {risk_score};
            const label = document.getElementById('gauge-label');

            function animate() {{
                cur = Math.min(cur + 1, target);
                drawGauge(cur);
                if(label) label.innerHTML = cur + '%';
                if(cur < target) {{
                    setTimeout(animate, 25);
                }}
            }}

            animate();
        </script>
    </body>
    </html>
    """

    components.html(gauge_html, height=280)

    # --------- Graph section -------
    # --------- Graph section with TWO SEPARATE CHARTS -------
    st.markdown(f"""<h5 style='margin:2rem 0 .3rem 0;color:#5647a3'>{T['graph_title']}</h5>
                    <div style='color:#bcbcbc; font-size:.99rem; margin-bottom:1.3rem;'>{T['graph_sub']}</div>""",
                unsafe_allow_html=True)

    # Create two columns for side-by-side charts
    chart_col1, chart_col2 = st.columns(2)

    x = np.arange(1, 13)
    user_glucose = np.full(len(x), float(avg_glucose))
    pop_glucose = np.clip(np.random.normal(115, 10, len(x)), 90, 140)
    user_bmi = np.full(len(x), float(bmi))
    pop_bmi = np.clip(np.random.normal(25, 3, len(x)), 18, 35)

    # CHART 1: Glucose Levels
    with chart_col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=x,
            y=user_glucose,
            mode='lines+markers',
            name='Your Glucose',
            line=dict(color='#5647a3', width=4),
            marker=dict(size=10, symbol='circle')
        ))
        fig1.add_trace(go.Scatter(
            x=x,
            y=pop_glucose,
            mode='lines',
            name='Population Avg',
            line=dict(color='#b0b0b0', width=3, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(176, 176, 176, 0.1)'
        ))

        fig1.update_layout(
            title=dict(
                text='<b>Average Glucose Levels</b>',
                font=dict(size=18, color='#5647a3', family='Montserrat'),
                x=0.5,
                xanchor='center'
            ),
            height=400,
            plot_bgcolor='#fafafa',
            paper_bgcolor='#fff',
            xaxis=dict(
                title='<b>Month</b>',
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=2,
                linecolor='#d0d0d0',
                dtick=1
            ),
            yaxis=dict(
                title='<b>Glucose (mg/dL)</b>',
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=2,
                linecolor='#d0d0d0',
                range=[max(0, min(user_glucose.min(), pop_glucose.min()) - 20),
                       max(user_glucose.max(), pop_glucose.max()) + 20]
            ),
            legend=dict(
                orientation='h',
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            font=dict(family='Montserrat', size=13),
            margin=dict(l=60, r=30, t=60, b=80),
            hovermode='x unified'
        )

        st.plotly_chart(fig1, use_container_width=True)

    # CHART 2: BMI
    with chart_col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x,
            y=user_bmi,
            mode='lines+markers',
            name='Your BMI',
            line=dict(color='#49db83', width=4),
            marker=dict(size=10, symbol='circle')
        ))
        fig2.add_trace(go.Scatter(
            x=x,
            y=pop_bmi,
            mode='lines',
            name='Population Avg',
            line=dict(color='#cccccc', width=3, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(204, 204, 204, 0.1)'
        ))

        fig2.update_layout(
            title=dict(
                text='<b>Body Mass Index (BMI)</b>',
                font=dict(size=18, color='#49db83', family='Montserrat'),
                x=0.5,
                xanchor='center'
            ),
            height=400,
            plot_bgcolor='#fafafa',
            paper_bgcolor='#fff',
            xaxis=dict(
                title='<b>Month</b>',
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=2,
                linecolor='#d0d0d0',
                dtick=1
            ),
            yaxis=dict(
                title='<b>BMI (kg/m²)</b>',
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=2,
                linecolor='#d0d0d0',
                range=[max(0, min(user_bmi.min(), pop_bmi.min()) - 5),
                       max(user_bmi.max(), pop_bmi.max()) + 5]
            ),
            legend=dict(
                orientation='h',
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            font=dict(family='Montserrat', size=13),
            margin=dict(l=60, r=30, t=60, b=80),
            hovermode='x unified'
        )

        st.plotly_chart(fig2, use_container_width=True)

    # --------- Warning and advice --------
    st.markdown(f'<div class="warning-section">{T["warning"]}</div>', unsafe_allow_html=True)

    advice = []
    if risk_level == 'Low':
        advice.append(T['advice_low'])
    elif risk_level == 'Medium':
        advice.append(T['advice_medium'])
    elif risk_level == 'High':
        advice.append(T['advice_high'])

    if hypertension == T['yes']:
        advice.append(T['advice_hypertension'])
    if smoker == T['smoke_current']:
        advice.append(T['advice_smoke'])
    if avg_glucose > 115:
        advice.append(T['advice_glucose'])
    if bmi > 30:
        advice.append(T['advice_bmi'])

    if advice:
        st.markdown("<div class='advice-section'><b style='font-size:1.15rem;color:#5647a3;'>Personalized advice:</b>"
                    + "<ul style='margin:10px 0 0 6px;color:#36367b;font-size:1.09rem;font-weight:600;'>"
                    + "".join(f"<li>{adv}</li>" for adv in advice) + "</ul></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
