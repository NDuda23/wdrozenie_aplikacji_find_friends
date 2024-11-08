import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model 
import plotly.express as px  # type: ignore
from qdrant_client import QdrantClient
from dotenv import dotenv_values

env = dotenv_values(".env")

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=env["QDRANT_URL"], 
    api_key=env["QDRANT_API_KEY"],
    )


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())


@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters



with st.sidebar:
    st.header(":writing_hand: Powiedz nam coś o sobie!")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają zainteresowania podobne do Twoich")
    age = st.selectbox(":birthday: Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox(":mortar_board: Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox(":cat: Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox(":world_map: Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio(":man_and_woman_holding_hands: Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()


predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]


st.header(f":male-detective: Najbliżej Ci do grupy o nazwie:\n{predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric(":book: Liczba twoich znajomych", len(same_cluster_df))

st.header(":thinking_face: Kim są osoby z Twojej grupy?")

fig = px.histogram(same_cluster_df.sort_values("age"), x="age", color="age", color_discrete_sequence=px.colors.qualitative.Set1 )
fig.update_layout(
    title="Wiek Twoich znajomych",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób"
)
st.plotly_chart(fig)

#fig = px.histogram(same_cluster_df, x="edu_level")
fig = px.histogram(same_cluster_df, x="edu_level", color="edu_level", color_discrete_sequence=px.colors.qualitative.Set1)
fig.update_layout(
    title="Wykształcenie Twoich znajomych",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals", color="fav_animals", color_discrete_sequence=px.colors.qualitative.Set1)
fig.update_layout(
    title="Ulubione zwierzęta Twoich znajomych",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place", color="fav_place", color_discrete_sequence=px.colors.qualitative.Set1)
fig.update_layout(
    title="Ulubione miejsca Twoich znajomych",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender", color="gender", color_discrete_sequence=px.colors.qualitative.Set1)
fig.update_layout(
    title="Płeć Twoich znajomych",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)


