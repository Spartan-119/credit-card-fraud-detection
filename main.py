import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow import keras
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, Embedding, Dropout,Input
from keras import Model
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


class DataHandler:
    def __init__(self, data):
        self.data = data
        self.x = data.iloc[:, : -1]

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

edited_df = st.experimental_data_editor(df, num_rows=1)
edited_x, edited_df = DataHandler(edited_df).reshape_3d().get_data()
prob = model.predict(edited_x)
st.text(f'Probability of being Fraud With LSTM model:{prob[0][0][0]:.5f}')
