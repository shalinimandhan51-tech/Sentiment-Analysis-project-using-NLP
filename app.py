import string
import joblib
import streamlit as st
import pandas as pd
import altair as alt

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Preprocessing — mirrors the exact steps used in the training notebook:
#   1. lowercase
#   2. remove punctuation
#   3. remove digits
#   4. remove non-ascii characters (emojis etc.)
#   5. remove stopwords
# Stopwords are the standard NLTK English list, hardcoded so the app doesn't
# need to download anything at runtime.
# ---------------------------------------------------------------------------
STOPWORDS = set(
    """i me my myself we our ours ourselves you you're you've you'll you'd your yours
    yourself yourselves he him his himself she she's her hers herself it it's its itself
    they them their theirs themselves what which who whom this that that'll these those
    am is are was were be been being have has had having do does did doing a an the and
    but if or because as until while of at by for with about against between into through
    during before after above below to from up down in out on off over under again further
    then once here there when where why how all any both each few more most other some such
    no nor not only own same so than too very s t can will just don don't should should've
    now d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn
    hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn
    needn't shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn
    wouldn't""".split()
)

LABEL_MAP = {0: "sadness", 1: "anger", 2: "love", 3: "surprise", 4: "fear", 5: "joy"}

EMOJI_MAP = {
    "sadness": "😢",
    "anger": "😡",
    "love": "❤️",
    "surprise": "😲",
    "fear": "😨",
    "joy": "😄",
}

COLOR_MAP = {
    "sadness": "#4C72B0",
    "anger": "#D62728",
    "love": "#E377C2",
    "surprise": "#FF9F1C",
    "fear": "#7F3FBF",
    "joy": "#2CA02C",
}


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    return "".join(ch for ch in text if not ch.isdigit())


def remove_non_ascii(text: str) -> str:
    return "".join(ch for ch in text if ch.isascii())


def remove_stopwords(text: str) -> str:
    return " ".join(word for word in text.split() if word not in STOPWORDS)


def preprocess(text: str) -> str:
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_non_ascii(text)
    text = remove_stopwords(text)
    return text


# ---------------------------------------------------------------------------
# Load model + vectorizer (cached so it only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_regression_tfidf_model (1).joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer


model, vectorizer = load_artifacts()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎭 Emotion Detector")
st.write(
    "Type a sentence and this app will predict the emotion behind it using a "
    "TF-IDF + Logistic Regression model trained on six emotions: "
    "**sadness, anger, love, surprise, fear, joy**."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app loads a scikit-learn `TfidfVectorizer` and a "
        "`LogisticRegression` classifier that were trained on a labeled "
        "emotion dataset. Your input text goes through the same cleaning "
        "steps used during training before being classified."
    )
    st.markdown("**Pipeline steps:**")
    st.markdown(
        "1. Lowercase\n"
        "2. Remove punctuation\n"
        "3. Remove numbers\n"
        "4. Remove non-ASCII characters\n"
        "5. Remove stopwords\n"
        "6. TF-IDF vectorize\n"
        "7. Logistic Regression predicts"
    )
    show_cleaned = st.checkbox("Show cleaned text sent to the model", value=False)

user_text = st.text_area(
    "Enter your text:",
    placeholder="e.g. I can't believe how wonderful today turned out to be!",
    height=120,
)

analyze_clicked = st.button("Analyze Emotion", type="primary", use_container_width=True)

if analyze_clicked:
    if not user_text or not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = preprocess(user_text)

        if not cleaned.strip():
            st.error(
                "After cleaning, there's no text left to analyze "
                "(it may have been all punctuation/stopwords). Try different input."
            )
        else:
            vector = vectorizer.transform([cleaned])
            pred_class = model.predict(vector)[0]
            pred_label = LABEL_MAP[pred_class]
            probabilities = model.predict_proba(vector)[0]

            st.markdown("---")
            st.subheader(f"Predicted Emotion: {EMOJI_MAP[pred_label]} **{pred_label.capitalize()}**")

            if show_cleaned:
                st.caption(f"Cleaned text: `{cleaned}`")

            # Build a dataframe of probabilities for the chart, in a fixed
            # emotion order for a stable/readable layout.
            prob_df = pd.DataFrame(
                {
                    "emotion": [LABEL_MAP[c] for c in model.classes_],
                    "probability": probabilities,
                }
            )
            prob_df["emotion_label"] = prob_df["emotion"].str.capitalize()
            prob_df = prob_df.sort_values("probability", ascending=False)

            chart = (
                alt.Chart(prob_df)
                .mark_bar()
                .encode(
                    x=alt.X("probability:Q", title="Confidence", axis=alt.Axis(format="%")),
                    y=alt.Y("emotion_label:N", sort="-x", title=None),
                    color=alt.Color(
                        "emotion:N",
                        scale=alt.Scale(
                            domain=list(COLOR_MAP.keys()),
                            range=list(COLOR_MAP.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("emotion_label:N", title="Emotion"),
                        alt.Tooltip("probability:Q", title="Confidence", format=".1%"),
                    ],
                )
                .properties(height=250)
            )
            st.altair_chart(chart, use_container_width=True)

            st.dataframe(
                prob_df[["emotion_label", "probability"]]
                .rename(columns={"emotion_label": "Emotion", "probability": "Confidence"})
                .assign(Confidence=lambda d: (d["Confidence"] * 100).round(2).astype(str) + "%")
                .reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

st.markdown("---")
st.caption(
    "Model: Logistic Regression trained on TF-IDF features. "
    "Note: predictions reflect patterns learned from the training data and "
    "may not always be accurate."
)
