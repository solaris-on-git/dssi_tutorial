import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def generate_random_data(num_samples, slope, intercept, noise_level):
    np.random.seed(42)
    X = np.random.rand(num_samples, 1)
    y = slope * X + intercept + np.random.randn(num_samples, 1) * noise_level
    return X, y

def linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    st.title('Linear Regression Prediction')

    st.sidebar.header('Model Parameters')
    slope = st.sidebar.slider('Slope', min_value=0.1, max_value=2.0, value=1.0)
    intercept = st.sidebar.slider('Intercept', min_value=-5.0, max_value=5.0, value=0.0)
    noise_level = st.sidebar.slider('Noise Level', min_value=0.0, max_value=2.0, value=0.5)

    num_samples = st.sidebar.number_input('Number of samples', min_value=10, max_value=100, value=50)

    # st.write('Generating Random Data...')
    X_train, y_train = generate_random_data(num_samples, slope, intercept, noise_level)

    # st.write('Training Linear Regression Model...')
    model = linear_regression(X_train, y_train)
    # st.write('Model trained successfully.')

    st.write('Enter X values for prediction:')
    num_samples_pred = st.number_input('Number of samples', min_value=1, step=1, value=1)

    X_input = []
    for i in range(num_samples_pred):
        X_input.append(st.number_input(f'X[{i}]', value=0.0))

    X_pred = np.array(X_input).reshape(-1, 1)

    st.write('Input X values:')
    st.write(X_pred)

    if st.button('Predict'):
        y_pred = model.predict(X_pred)
        st.write('Predicted y values:')
        st.write(y_pred)

if __name__ == '__main__':
    main()
