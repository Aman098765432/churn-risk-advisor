
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Churn Risk Advisor",
    layout="wide"
)

# ---------------- COLUMNS ----------------

RAW_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

SERVICES = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

PAYMENTS = [
    'Electronic check',
    'Mailed check',
    'Bank transfer (automatic)',
    'Credit card (automatic)'
]

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.joblib')

    with open('model_meta.json', 'r') as f:
        meta = json.load(f)

    return model, meta


def prepare_input(raw, columns):
    X = pd.get_dummies(raw[RAW_COLS])

    return X.reindex(
        columns=columns,
        fill_value=0
    ).astype(float)


model, meta = load_artifacts()
COLS = meta['feature_columns']


def score(raw):
    prepared = prepare_input(raw, COLS)
    return model.predict_proba(prepared)[:, 1]


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header('Customer profile')

    tenure = st.slider(
        'Tenure (months)',
        0, 72, 4
    )

    contract = st.selectbox(
        'Contract',
        ['Month-to-month', 'One year', 'Two year']
    )

    monthly = st.number_input(
        'Monthly charges (USD)',
        min_value=18.0,
        max_value=120.0,
        value=85.0,
        step=0.5
    )

    internet = st.selectbox(
        'Internet service',
        ['Fiber optic', 'DSL', 'No']
    )

    services = []

    if internet != 'No':
        services = st.multiselect(
            'Add-on services',
            SERVICES,
            default=['StreamingTV']
        )

    phone = st.radio(
        'Phone service',
        ['Yes', 'No'],
        horizontal=True
    )

    multi = 'No phone service'

    if phone == 'Yes':
        multi = st.radio(
            'Multiple lines',
            ['No', 'Yes'],
            horizontal=True
        )

    payment = st.selectbox(
        'Payment method',
        PAYMENTS
    )

    paperless = st.radio(
        'Paperless billing',
        ['Yes', 'No'],
        horizontal=True
    )

    with st.expander('Demographics'):

        gender = st.radio(
            'Gender',
            ['Female', 'Male'],
            horizontal=True
        )

        senior = st.checkbox('Senior citizen')
        partner = st.checkbox('Has partner')
        dependents = st.checkbox('Has dependents')

    st.divider()

    threshold = st.slider(
        'Contact threshold',
        min_value=0.05,
        max_value=0.95,
        value=float(meta['threshold']),
        step=0.05
    )


# ---------------- CUSTOMER DATA ----------------

row = {
    'gender': gender,
    'SeniorCitizen': int(senior),
    'Partner': 'Yes' if partner else 'No',
    'Dependents': 'Yes' if dependents else 'No',
    'tenure': tenure,
    'PhoneService': phone,
    'MultipleLines': multi,
    'InternetService': internet,
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly,
    'TotalCharges': tenure * monthly
}

for s in SERVICES:
    if internet == 'No':
        row[s] = 'No internet service'
    elif s in services:
        row[s] = 'Yes'
    else:
        row[s] = 'No'

customer = pd.DataFrame([row])


# ---------------- MAIN AREA ----------------

st.title('Customer Churn Risk Advisor')

st.caption(
    f"{meta['model_name']} | "
    f"CV AUC {meta['cv_auc']:.3f} "
    f"+/- {meta['cv_auc_std']:.3f} | "
    "Decision support only"
)

tab1, tab2, tab3 = st.tabs([
    'One customer',
    'Batch scoring',
    'About'
])


# ---------------- TAB 1: ONE CUSTOMER ----------------

with tab1:

    p = float(score(customer)[0])

    if p >= threshold:
        band = 'HIGH'
    elif p >= threshold / 2:
        band = 'WATCH'
    else:
        band = 'LOW'

    c1, c2, c3 = st.columns(3)

    c1.metric(
        'Churn probability',
        f'{p:.0%}'
    )

    c2.metric(
        'Risk band',
        band
    )

    c3.metric(
        'Action',
        'Contact now' if band == 'HIGH'
        else 'No action'
    )

    st.progress(p)

    # WHAT-IF ANALYSIS

    st.subheader('What would change the risk?')

    changes = [
        ('Contract', c)
        for c in ['One year', 'Two year']
        if c != contract
    ]

    if payment == 'Electronic check':
        changes.append(
            ('PaymentMethod', 'Credit card (automatic)')
        )

    if internet != 'No' and 'TechSupport' not in services:
        changes.append(
            ('TechSupport', 'Yes')
        )

    rows = []

    for feature, value in changes:

        alt = customer.copy()
        alt[feature] = value

        q = float(score(alt)[0])

        rows.append({
            'Change': f'{feature}: {value}',
            'New probability': round(q, 3),
            'Difference': round(q - p, 3)
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True
        )

    st.caption(
        'Associations learned from data, '
        'not guaranteed effects.'
    )

    # LOGISTIC REGRESSION EXPLANATION
    # Only visible if the model is Logistic Regression

    final = (
        model[-1]
        if hasattr(model, 'steps')
        else model
    )

    if hasattr(model, 'steps') and hasattr(final, 'coef_'):

        with st.expander(
            'Why this score? (Logistic Regression)'
        ):

            z = model[:-1].transform(
                prepare_input(customer, COLS)
            )[0]

            contrib = pd.Series(
                final.coef_[0] * z,
                index=COLS
            )

            top = contrib.reindex(
                contrib.abs()
                .sort_values(ascending=False)
                .index
            ).head(8)

            st.bar_chart(top)

            st.caption(
                'Positive bars push toward churn, '
                'negative away.'
            )


# ---------------- TAB 2: BATCH SCORING ----------------

with tab2:

    st.write(
        'Upload a CSV with the original Telco columns.'
    )

    file = st.file_uploader(
        'Customer file',
        type='csv'
    )

    if file is not None:

        data = pd.read_csv(file)

        missing = [
            c for c in RAW_COLS
            if c not in data.columns
        ]

        if missing:
            st.error(
                f'Missing columns: {missing}'
            )
            st.stop()

        data['TotalCharges'] = pd.to_numeric(
            data['TotalCharges'],
            errors='coerce'
        ).fillna(0)

        data['p_churn'] = score(data).round(3)

        data['contact'] = np.where(
            data['p_churn'] >= threshold,
            'Yes',
            ''
        )

        contact_count = (
            data['contact'] == 'Yes'
        ).sum()

        st.write(
            f"{contact_count} of {len(data)} "
            "customers are above the threshold."
        )

        st.dataframe(
            data.sort_values(
                'p_churn',
                ascending=False
            ).head(50)
        )

        st.download_button(
            'Download scored CSV',
            data.to_csv(index=False),
            'scored_customers.csv',
            'text/csv'
        )


# ---------------- TAB 3: ABOUT ----------------

with tab3:

    st.json({
        k: v
        for k, v in meta.items()
        if k != 'feature_columns'
    })

    st.markdown(
        '**Limitations:** trained on one US '
        'telecom dataset; not validated for '
        'other markets. See the model card.'
    )
