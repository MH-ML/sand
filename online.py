import pickle
from io import BytesIO

import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="Dredging Sand Matching Between Buyer And Supplier", page_icon=":building_construction:")

@st.cache_resource
def download_pickle_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        raise Exception(f"Failed to download pickle file from {url}")
    
    
# Load the dredging model and data
dredging_model_url = 'https://github.com/MH-ML/Row-Material/raw/main/dridging_model1.pkl'
dredging_model = download_pickle_from_github(dredging_model_url)

dredging_X_supplier_url = 'https://github.com/MH-ML/Row-Material/raw/main/dridging_supplier1.pkl'
dredging_X_supplier = download_pickle_from_github(dredging_X_supplier_url)

    
# Define function for prediction

def predict_dredging_match(buyer_price, buyer_availability, buyer_fm, num_predictions=4):
    predictions = []
    suppliers = []

    for _ in range(num_predictions):
        # Get a random supplier
        supplier = dredging_X_supplier.sample(1).squeeze().to_dict()

        # Make prediction
        prediction_input = np.array([[buyer_price, buyer_availability, buyer_fm,
                                       supplier['Supplier_Price'], supplier['Supplier_Availability'], supplier['FM_y']]])
        prediction = dredging_model.predict(prediction_input)
        predictions.append(prediction[0])
        suppliers.append(supplier)

    return predictions, suppliers

# Streamlit UI
def main():

    st.title("Dredging Sand Matching Between Buyer And Supplier")

    # Input fields for user
    st.sidebar.header("Buyer Details")
    buyer_details = {
        "Price": st.sidebar.number_input("Price"),
        "Availability": st.sidebar.number_input("Availability"),
        "FM": st.sidebar.number_input("FM")
    }

    # Prediction button
    if st.sidebar.button("Predict"):
        with st.spinner('Generating Results...'):
            predictions, suppliers = predict_dredging_match(
                buyer_details["Price"],
                buyer_details["Availability"],
                buyer_details["FM"]
            )

            st.write("---")

            st.subheader("Dashboard:")

            for i, (prediction, supplier) in enumerate(zip(predictions, suppliers), start=1):
                st.write(f"Prediction {i}:")
                if prediction >= 0.63:
                    st.success(f"Match Score: {prediction:.2f}%")
                    st.subheader("Buyer Details:")
                    st.info("Price: {}\nAvailability: {}\nFM: {}".format(
                        buyer_details["Price"],
                        buyer_details["Availability"],
                        buyer_details["FM"]
                    ))

                    st.subheader("Supplier Details:")
                    st.success("Supplier Price: {}\nSupplier Availability: {}\nSupplier FM: {}".format(
                        supplier['Supplier_Price'],
                        supplier['Supplier_Availability'],
                        supplier['FM_y']
                    ))

                    st.subheader("Overall Quality Match:")
                    st.success(f"Overall Quality Match between Buyer and Supplier: {prediction:.2f}%")

                else:
                    st.error(f"No match found. Match Score: {prediction:.2f}%")

                st.write("---")

if __name__ == "__main__":
    main()