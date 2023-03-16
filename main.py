import streamlit as st
from tensorflow import keras
import pandas as pd


class DataHandler:
    def __init__(self, data):
        self.data = data
        self.x = data.iloc[:, : -1]

    def drop_columns(self, columns):
        self.data = self.data.drop(columns, axis=1)
        self.x = self.x.drop(columns, axis=1)
        return self

    def reshape_3d_lstm(self):
        self.x = self.x.values.reshape((self.x.shape[0], 1, self.x.shape[1]))
        return self
    def reshape_3d_cnn(self):
        self.x = self.x.values.reshape((self.x.shape[0], self.x.shape[1], 1))
        return self
    def get_data(self):
        return self.x, self.data



is_showing_data = st.sidebar.button('Data')
is_showing_models = st.sidebar.button('Models')
is_testing = st.sidebar.button('Testing')
DATE_COLUMN = 'Amount'
DATA_URL = 'https://drive.google.com/uc?id=1aJJOGOT-9iKKnmvF-Rg5kl2BBWLm21Nx'

if 'last_button' not in st.session_state:
    st.session_state['last_button'] = 'def'
last_button = st.session_state['last_button']
@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    return data


data_load_state = st.text('Loading data...')
origin_data = load_data(100)
data_load_state.text("")
print(last_button)
if is_showing_data:

    st.subheader('Sample Data')
    st.write(origin_data)
    st.session_state['last_button'] = "data"

elif is_showing_models:
    st.subheader('Models')
    st.text('1. CNN Model:')
    st.text('2. Auto Encoder:')
    st.text('3. LSTM Model:')
    st.text('4. Atention Model:')
    st.session_state['last_button'] = "models"


elif is_testing or last_button == "testing":
    st.subheader('Input your Test data:')
    df = pd.DataFrame(
        [
            origin_data.iloc[0]
        ]
    )
    df_unclassed = df.drop(["Class"], axis=1)
    cols_to_delete = ['Time (second)', 'V5', 'V6', 'V7', 'V8', 'V9', 'V13', 'V15', 'V16', 'V18', 'V19', 'V20', 'V21',
                      'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
    edited_df = st.experimental_data_editor(df_unclassed, num_rows=1)
    edited_df["Class"] = 0
    edited_x_lstm, edited_df_lstm = DataHandler(edited_df).drop_columns(cols_to_delete).reshape_3d_lstm().get_data()
    edited_x_cnn, edited_df_cnn = DataHandler(edited_df).reshape_3d_cnn().get_data()
    edited_x_enc, edited_df_enc = DataHandler(edited_df).get_data()

    model_lstm = keras.models.load_model('lstm_saved_model.h5')
    prob_lstm = model_lstm.predict(edited_x_lstm)

    cnn_model = keras.models.load_model('cnn_saved_model.h5')
    prob_cnn = cnn_model.predict(edited_x_cnn)

    enc_model = keras.models.load_model('auto_saved_model.h5')
    prob_enc = enc_model.predict(edited_x_enc)
    print(prob_enc)

    probs_dict = {
        "Prob": {"CNN": prob_cnn[0][0],
            "Auto Encoder": prob_enc[0][0],
            "LSTM": prob_lstm[0][0][0]
            # "GAN": prob_cnn[0][0]
            }
    }
    chart_data = pd.DataFrame.from_dict(probs_dict)
    print(chart_data)

    st.bar_chart(chart_data)
    st.session_state['last_button'] = "testing"

else:
    st.title('Fraud Detection Using Deep Learning')

    st.subheader('A comparison between different Deep learning models by: Muskan Asmath, Abin Varghese, Neha Gupta, And Amirali Monjar')
    st.text('In this project we trained and evaluated four different deep learning models \non '
            'Credit Card fraud detection dataset (2013).\n'
            'These four models are based on CNN, AutoEncoders, LSTM, and GAN.\n'
               'Use the sidebar on left to navigate.')



