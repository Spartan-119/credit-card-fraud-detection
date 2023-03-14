import streamlit as st
from tensorflow import keras
import pandas as pd


class DataHandler:
    def __init__(self, data):
        self.data = data
        self.x = data.iloc[:, : -1]

    def drop_columns(self, columns):
        self.data = self.data.drop(columns,axis =1)
        self.x = self.x.drop(columns,axis =1)
        return self
    def reshape_3d(self):
        self.x = self.x.values.reshape((self.x.shape[0], 1, self.x.shape[1]))
        return self

    def get_data(self):
        return self.x, self.data


st.title('Uber pickups in NYC')

DATE_COLUMN = 'Amount'
DATA_URL = 'https://drive.google.com/uc?id=1aJJOGOT-9iKKnmvF-Rg5kl2BBWLm21Nx'


@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    return data


data_load_state = st.text('Loading data...')
origin_data = load_data(100)

if st.checkbox("Show Sample Data"):
    st.subheader('Sample Data')
    st.write(origin_data)
data_load_state.text("")
st.subheader('Input your Test data:')
model = keras.models.load_model('lstm_saved_model.h5')
df = pd.DataFrame(
    [
        origin_data.iloc[0]
    ]
)
cols_to_delete = ['Time (second)', 'V5', 'V6', 'V7', 'V8', 'V9','V13','V15', 'V16',  'V18', 'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
edited_df = st.experimental_data_editor(df, num_rows=1)
edited_x, edited_df = DataHandler(edited_df).drop_columns(cols_to_delete).reshape_3d().get_data()
prob = model.predict(edited_x)
st.text(f'Probability of being Fraud With LSTM model:{prob[0][0][0]:.5f}')
