import streamlit as st
#import pandas as pd
import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error



st.title('Stock Price Predictions')


def main():
    #user_input = st.selectbox('Choose', [ 'Predict'])
    predict()



#@st.cache_resource
def download_data(user_input, start_date, end_date):
    df = yf.download(user_input, start=start_date, end=end_date, progress=False)
    return df


option = st.selectbox('Enter Currency Ticker', ['EURUSD=X','JPY=X', 'GBPUSD=X'])
option = option.upper()
today = datetime.date.today()
start = today - datetime.timedelta(days=3650)
start_date = start
end_date = today





data = download_data(option, start_date, end_date)
scaler = StandardScaler()


st.write('Close Price')
st.line_chart(data.Close)



def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))



def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'KNeighborsRegressor'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)



def model_engine(model, num): 
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df.preds.values
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


if __name__ == '__main__':
    main()