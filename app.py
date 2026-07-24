from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
import streamlit as st

# ---------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE & STYLES CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Academia ML | Détection de Fraudes",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        color: white !important;
    }
    [data-testid="stSidebar"] * { color: #f8fafc !important; }

    /* CADRE HEURE & DATE EN HAUT */
    .time-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .time-card .date-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #38bdf8 !important;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .time-card .time-display {
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff !important;
        font-family: 'Courier New', Courier, monospace;
        letter-spacing: 2px;
    }
    .time-card .date-display {
        font-size: 0.85rem;
        color: #94a3b8 !important;
        font-weight: 500;
        margin-top: 2px;
    }

    /* EN-TÊTE SIDEBAR ACADEMIA */
    .academia-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .academia-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 1px;
        color: #f1f5f9 !important;
        margin-bottom: 2px;
    }
    .academia-subtitle {
        font-size: 0.8rem;
        color: #38bdf8 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }

    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-card h4 {
        color: #475569;
        margin-bottom: 5px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card p {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #f1f5f9;
        border-radius: 8px;
        color: #334155;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 2. DÉFINITION DES VARIABLES & PIPELINE DE PRÉPARATION
# ---------------------------------------------------------

EXACT_FEATURE_COLUMNS = [
    'age',
    'salaire',
    'score_credit',
    'montant_transaction',
    'anciennete_compte',
    'type_carte_Visa',
    'region_Miami',
    'region_Orlando',
    'genre_male',
]

CARTE_OPTIONS = ['Mastercard', 'Visa']
REGION_OPTIONS = ['Houston', 'Miami', 'Orlando']
GENRE_OPTIONS = ['femelle', 'male']


def prepare_input_features(
    age,
    salaire,
    score_credit,
    montant,
    anciennete,
    type_carte,
    region,
    genre,
    scaler,
):
  # 1. Construction du dictionnaire avec les 9 features
  raw_data = {
      'age': float(age),
      'salaire': float(salaire),
      'score_credit': float(score_credit),
      'montant_transaction': float(montant),
      'anciennete_compte': float(anciennete),
      'type_carte_Visa': 1 if type_carte == 'Visa' else 0,
      'region_Miami': 1 if region == 'Miami' else 0,
      'region_Orlando': 1 if region == 'Orlando' else 0,
      'genre_male': 1 if genre == 'male' else 0,
  }

  df_raw = pd.DataFrame([raw_data])[EXACT_FEATURE_COLUMNS]

  # 2. Transformation des 9 variables par le scaler
  scaled_array = scaler.transform(df_raw)

  # 3. Retourne le DataFrame prêt pour le modèle KNN
  return pd.DataFrame(scaled_array, columns=EXACT_FEATURE_COLUMNS)


# ---------------------------------------------------------
# 3. CHARGEMENT SÉCURISÉ DES ARTEFACTS ET DES DONNÉES
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
  model = joblib.load('modele_knn.pkl')
  try:
    scaler = joblib.load('scaler_final.pkl')
  except Exception:
    scaler = joblib.load('scaler.pkl')
  return model, scaler


@st.cache_data
def load_cleaned_data():
  try:
    # Tente d'abord de charger depuis donnees_preparees.pkl
    data = joblib.load('donnees_preparees.pkl')
    if isinstance(data, dict) and 'df_cleaned' in data:
      return data['df_cleaned']
    # Sinon tente le CSV
    return pd.read_csv('df_cleaned.csv')
  except Exception:
    try:
      return pd.read_csv('df_cleaned.csv')
    except Exception:
      return None


@st.cache_data
def load_prepared_data():
  try:
    data = joblib.load('donnees_preparees.pkl')
    return data.get('X_test_scaled'), data.get('y_test')
  except Exception:
    return None, None


try:
  model, scaler = load_artifacts()
  status_ok = True
except Exception as e:
  status_ok = False
  st.error(f'⚠️ Erreur de chargement des artefacts ML : {e}')

df_cleaned = load_cleaned_data()
X_test_scaled, y_test = load_prepared_data()

# ---------------------------------------------------------
# 4. SIDEBAR REDISPOSÉE & STYLE ACADEMIA
# ---------------------------------------------------------
with st.sidebar:
  # 1. HORLOGE SYSTEME EN HAUT
  now = datetime.now()
  date_str = now.strftime('%A %d %B %Y').capitalize()
  time_str = now.strftime('%H:%M:%S')

  st.markdown(
      f"""
    <div class="time-card">
        <div class="date-title">⏱️ HORODATAGE SYSTÈME</div>
        <div class="time-display">{time_str}</div>
        <div class="date-display">{date_str}</div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  # 2. TITRE ACADEMIA
  st.markdown(
      """
    <div class="academia-header">
        <div class="academia-title">🎓 ACADEMIA ML</div>
        <div class="academia-subtitle">Détection & Analyse Anti-Fraude</div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  st.divider()

  # 3. ÉTAT DU MOTEUR
  if status_ok:
    st.success('🟢 Moteur KNN Opérationnel')
  else:
    st.error('🔴 Moteur Hors-Ligne')

  # 4. SEUIL DE DECISION
  st.markdown('### ⚙️ Sensibilité du Filtre')
  threshold = st.slider(
      'Seuil de décision (%)', min_value=10, max_value=90, value=50, step=5
  )

  st.divider()

  # 5. ÉTAT DES FICHIERS
  st.markdown('📊 **Dataset Nettoyé :**')
  if df_cleaned is not None:
    st.caption(f'✓ {len(df_cleaned)} lignes chargées')
  else:
    st.caption('⚠️ Dataset non trouvé')

  st.markdown('🧪 **Données de Test :**')
  if X_test_scaled is not None:
    st.caption(f'✓ {len(X_test_scaled)} exemples prêts')
  else:
    st.caption('⚠️ `donnees_preparees.pkl` non trouvé')

# ---------------------------------------------------------
# 5. EN-TÊTE PRINCIPAL
# ---------------------------------------------------------
st.markdown(
    '<p class="main-header">🎓 ACADEMIA ML — Détection Algorithmique des'
    ' Fraudes</p>',
    unsafe_allow_html=True,
)
st.markdown(
    'Plateforme d\'évaluation continue basée sur un modèle **K-Nearest Neighbors'
    ' (KNN)** avec équilibrage **SMOTE-Tomek**.'
)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '🔮 Prédiction Unitaire',
    '📁 Analyse Batch (CSV)',
    '🗃️ Dataset Nettoyé',
    '📊 Diagnostics & Performance',
    '💰 Impact Financier & ROI',
])

# ---------------------------------------------------------
# ONGLET 1 : PREDICTION UNITAIRE
# ---------------------------------------------------------
with tab1:
  st.subheader("📋 Évaluation d'une Transaction en Temps Réel")

  col_in1, col_in2 = st.columns(2)

  with col_in1:
    st.markdown('##### 👤 Profil Client')
    age = st.number_input(
        'Âge du titulaire', min_value=18, max_value=100, value=38
    )
    salaire = st.number_input(
        'Revenu mensuel (FCFA)', min_value=0, value=450000, step=10000
    )
    score_credit = st.slider(
        'Score de Crédit (300-850)', min_value=300, max_value=850, value=680
    )
    genre = st.selectbox('Genre', GENRE_OPTIONS)

  with col_in2:
    st.markdown('##### 💳 Détails Transaction')
    montant = st.number_input(
        'Montant de la transaction (FCFA)',
        min_value=0,
        value=850000,
        step=5000,
    )
    anciennete = st.slider(
        'Ancienneté du compte (années)', min_value=0, max_value=30, value=4
    )
    type_carte = st.selectbox('Type de carte', CARTE_OPTIONS)
    region = st.selectbox('Région', REGION_OPTIONS)

  st.markdown('<br>', unsafe_allow_html=True)
  btn_predict = st.button("🔍 Lancer l'Analyse de Risque")

  if btn_predict and status_ok:
    input_prepared = prepare_input_features(
        age,
        salaire,
        score_credit,
        montant,
        anciennete,
        type_carte,
        region,
        genre,
        scaler,
    )

    probabilities = model.predict_proba(input_prepared)[0]
    prob_fraude = float(probabilities[1]) * 100
    is_fraude = prob_fraude >= threshold

    st.divider()
    res_col1, res_col2 = st.columns([1, 1.5])

    with res_col1:
      fig_gauge = go.Figure(
          go.Indicator(
              mode='gauge+number',
              value=prob_fraude,
              number={'suffix': '%', 'valueformat': '.1f'},
              title={'text': 'Score de Risque de Fraude'},
              gauge={
                  'axis': {'range': [0, 100]},
                  'bar': {'color': '#0f172a'},
                  'steps': [
                      {'range': [0, 30], 'color': '#86efac'},
                      {'range': [30, 70], 'color': '#fde047'},
                      {'range': [70, 100], 'color': '#f87171'},
                  ],
                  'threshold': {
                      'line': {'color': 'red', 'width': 4},
                      'thickness': 0.75,
                      'value': threshold,
                  },
              },
          )
      )
      fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
      st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
      if is_fraude:
        st.error(
            '🚨 **SUSPICION DE FRAUDE ÉLEVÉE (Risque:'
            f' {prob_fraude:.1f}%)**'
        )
        st.markdown(
            '* **Action Recommandée :** Blocage immédiat de la transaction.'
        )
        st.markdown(
            '* **Contrôle :** Transmission au pôle de vérification d\'identité.'
        )
      else:
        st.success(
            f'🟢 **TRANSACTION CONFORME (Risque: {prob_fraude:.1f}%)**'
        )
        st.markdown(
            '* **Action Recommandée :** Traitement automatique de la'
            ' transaction.'
        )

# ---------------------------------------------------------
# ONGLET 2 : BATCH PREDICTION (CSV)
# ---------------------------------------------------------
with tab2:
  st.subheader('📁 Traitement Automatisé par Fichier CSV')
  uploaded_file = st.file_uploader(
      'Importer un fichier de transactions à auditer', type=['csv']
  )

  if uploaded_file is not None and status_ok:
    batch_data = pd.read_csv(uploaded_file)
    st.write('Aperçu du fichier :', batch_data.head(3))

    raw_cols = [
        'age',
        'salaire',
        'score_credit',
        'montant_transaction',
        'anciennete_compte',
        'type_carte',
        'region',
        'genre',
    ]
    missing_cols = [c for c in raw_cols if c not in batch_data.columns]

    if missing_cols:
      st.error(f'⚠️ Colonnes manquantes dans le CSV : {missing_cols}')
    else:
      if st.button('🚀 Auditer le Fichier Batch'):
        encoded_list = []
        for idx, row in batch_data.iterrows():
          enc_row = prepare_input_features(
              row['age'],
              row['salaire'],
              row['score_credit'],
              row['montant_transaction'],
              row['anciennete_compte'],
              row['type_carte'],
              row['region'],
              row['genre'],
              scaler,
          )
          encoded_list.append(enc_row)

        encoded_batch = pd.concat(encoded_list, ignore_index=True)
        probs = model.predict_proba(encoded_batch)[:, 1] * 100
        predictions = (probs >= threshold).astype(int)

        batch_data['Risque_%'] = np.round(probs, 2)
        batch_data['Verdict'] = np.where(
            predictions == 1, '🚨 Fraude', '🟢 Légitime'
        )

        st.success(f'✅ Audit terminé sur {len(batch_data)} transactions.')
        st.dataframe(batch_data, use_container_width=True)

        c_m1, c_m2 = st.columns(2)
        c_m1.metric('Nombre total de transactions', len(batch_data))
        c_m2.metric('Fraudes suspectées', int(predictions.sum()))

# ---------------------------------------------------------
# ONGLET 3 : EXPLORATION DU DATAFRAME NETTOYÉ
# ---------------------------------------------------------
with tab3:
  st.subheader('🗃️ Données Nettoyées')

  if df_cleaned is not None:
    st.dataframe(df_cleaned, use_container_width=True)

    col_g1, col_g2 = st.columns(2)
    if 'region' in df_cleaned.columns and 'fraude' in df_cleaned.columns:
      with col_g1:
        taux_region = (
            df_cleaned.groupby('region')['fraude'].mean().reset_index()
        )
        taux_region['fraude'] *= 100
        fig_region = px.bar(
            taux_region,
            x='region',
            y='fraude',
            title='Taux de fraude par région (%)',
            color_discrete_sequence=['#2563eb'],
        )
        st.plotly_chart(fig_region, use_container_width=True)

    if 'type_carte' in df_cleaned.columns and 'fraude' in df_cleaned.columns:
      with col_g2:
        taux_carte = (
            df_cleaned.groupby('type_carte')['fraude'].mean().reset_index()
        )
        taux_carte['fraude'] *= 100
        fig_carte = px.bar(
            taux_carte,
            x='type_carte',
            y='fraude',
            title='Taux de fraude par type de carte (%)',
            color_discrete_sequence=['#0f172a'],
        )
        st.plotly_chart(fig_carte, use_container_width=True)
  else:
    st.warning('⚠️ Fichier nettoyé introuvable.')

# ---------------------------------------------------------
# ONGLET 4 : DIAGNOSTICS & DASHBOARD
# ---------------------------------------------------------
with tab4:
  st.subheader('📊 Diagnostics de Modélisation (Données de Test)')

  c1, c2, c3 = st.columns(3)
  c1.markdown(
      '<div class="metric-card"><h4>Algorithme</h4><p>KNN'
      ' Optimisé</p></div>',
      unsafe_allow_html=True,
  )
  c2.markdown(
      '<div class="metric-card"><h4>Rééchantillonnage</h4><p>SMOTE-Tomek</p></div>',
      unsafe_allow_html=True,
  )
  c3.markdown(
      '<div class="metric-card"><h4>Standardisation</h4><p>StandardScaler</p></div>',
      unsafe_allow_html=True,
  )

  st.markdown('<br>', unsafe_allow_html=True)

  if status_ok and X_test_scaled is not None and y_test is not None:
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Accuracy', f'{accuracy_score(y_test, y_pred)*100:.1f}%')
    m2.metric(
        'Precision',
        f'{precision_score(y_test, y_pred, zero_division=0)*100:.1f}%',
    )
    m3.metric(
        'Recall', f'{recall_score(y_test, y_pred, zero_division=0)*100:.1f}%'
    )
    m4.metric(
        'F1-score', f'{f1_score(y_test, y_pred, zero_division=0)*100:.1f}%'
    )

    st.divider()
    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
      cm = confusion_matrix(y_test, y_pred)
      fig_cm = px.imshow(
          cm,
          text_auto=True,
          labels=dict(x='Prédit', y='Réel', color='Nombre'),
          x=['Légitime', 'Fraude'],
          y=['Légitime', 'Fraude'],
          color_continuous_scale='Blues',
          title='Matrice de Confusion',
      )
      st.plotly_chart(fig_cm, use_container_width=True)

    with diag_col2:
      fpr, tpr, _ = roc_curve(y_test, y_proba)
      roc_auc = auc(fpr, tpr)
      fig_roc = go.Figure()
      fig_roc.add_trace(
          go.Scatter(
              x=fpr,
              y=tpr,
              mode='lines',
              name=f'ROC (AUC = {roc_auc:.3f})',
              line=dict(color='#2563eb', width=3),
          )
      )
      fig_roc.add_trace(
          go.Scatter(
              x=[0, 1],
              y=[0, 1],
              mode='lines',
              name='Hasard',
              line=dict(dash='dash', color='gray'),
          )
      )
      fig_roc.update_layout(
          title='Courbe ROC',
          xaxis_title='Taux Faux Positifs',
          yaxis_title='Taux Vrais Positifs',
      )
      st.plotly_chart(fig_roc, use_container_width=True)

    with st.expander('🔍 Importance des variables (Permutation Importance)'):
      perm = permutation_importance(
          model, X_test_scaled, y_test, n_repeats=10, random_state=42
      )
      importance_df = pd.DataFrame({
          'Variable': EXACT_FEATURE_COLUMNS,
          'Importance': perm.importances_mean,
      }).sort_values('Importance', ascending=True)
      fig_imp = px.bar(
          importance_df,
          x='Importance',
          y='Variable',
          orientation='h',
          color_discrete_sequence=['#0f172a'],
      )
      st.plotly_chart(fig_imp, use_container_width=True)
  else:
    st.warning('⚠️ Exportez `donnees_preparees.pkl` pour afficher ces métriques.')

# ---------------------------------------------------------
# ONGLET 5 : ROI
# ---------------------------------------------------------
with tab5:
  st.subheader("💡 Calculateur d'Impact Financier (ROI)")
  col_r1, col_r2 = st.columns(2)
  with col_r1:
    v_transac = st.number_input(
        'Volume Annuel de Transactions', value=100000, step=10000
    )
    c_fraude = st.number_input(
        "Coût Moyen d'une Fraude (FCFA)", value=250000, step=25000
    )
  with col_r2:
    t_fraude = st.slider('Taux de Fraude Estimé (%)', 0.1, 5.0, 1.0)
    taux_detection = st.slider('Recall du Modèle (%)', 10, 100, 70)

  total_fraudes = int(v_transac * (t_fraude / 100))
  exposition = total_fraudes * c_fraude
  economie = int(exposition * (taux_detection / 100))

  st.divider()
  m_r1, m_r2 = st.columns(2)
  m_r1.metric('Pertes Totales Sans Modèle', f'{exposition:,} FCFA')
  m_r2.metric('Économies Réalisées grâce au ML', f'{economie:,} FCFA')
