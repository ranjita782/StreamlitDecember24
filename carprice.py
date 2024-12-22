
import streamlit as st
import pandas as pd
import pickle
import sklearn

cars_df = pd.read_csv("./cars24-car-price.csv")

st.title("Car Resale Price Prediction")
st.dataframe(cars_df.head())

with open('car_pred_model', 'rb') as f:
    model = pickle.load(f)

# sklearn model which is trained on cars24 data.



col1, col2 = st.columns(2)

# dropdown
fuel_type = col1.selectbox("Select the fuel type",
                           ["Diesel", "Petrol", "CNG", "LPG", "Electric"])

engine = col1.slider("Set the Engine Power",
                     500, 5000, step=100)

transmission_type = col2.selectbox("Select the transmission type",
                                   ["Manual", "Automatic"])

seats = col2.selectbox("Enter the number of seats",
                       [4,5,7,9,11])


## Encoding Categorical features
## Use the same encoding as used during the training. 
encode_dict = {
    "fuel_type": {'Diesel': 1, 'Petrol': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5},
    "seller_type": {'Dealer': 1, 'Individual': 2, 'Trustmark Dealer': 3},
    "transmission_type": {'Manual': 1, 'Automatic': 2}
}


if st.button("Get Price"):
    # predict here

    encoded_fuel_type = encode_dict['fuel_type'][fuel_type]
    encoded_transmission_type = encode_dict['transmission_type'][transmission_type]

    input_car = [2012.0,2,120000,encoded_fuel_type,encoded_transmission_type,19.7,engine,46.3,seats]
    price = model.predict([input_car])[0]

    st.header(round(price,2))







# scaler = StandardScaler()
# scaler.fit_transform(X)

# pickle.dump(scaler)

# scaler.transform()
