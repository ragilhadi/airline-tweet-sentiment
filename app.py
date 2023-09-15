import streamlit as st
import pickle
from utils import preprocessing_text

st.set_page_config(
    page_title="Airplane Tweet Sentiment Analysis",
    page_icon=":rocket:"
)

if 'model' not in st.session_state:
    model = pickle.load(open('model/model.sav', 'rb'))
    st.session_state['model'] = model

st.title('Air Passanger Tweets Sentiment Analysis')

tweet = st.text_area('Text to analyze', 
                     "@VirginAmerica and it's a really big bad thing about it")

if st.button('Analyze'):
    tweet_prep = preprocessing_text(tweet)
    result = st.session_state['model'].predict([tweet_prep])
    if result.tolist()[0] == 0:
        result = 'Neutral or Positive Tweet'
    else:
        result = 'Negative Tweet'
    st.write('Sentiment:', result)
else:
    st.write('waiting')
