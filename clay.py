import pickle
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from scipy.special import expit  # Import the expit function (sigmoid function)

st.set_page_config(page_title="Red Clay Brick Matching Between Buyer And Supplier", page_icon=":building_construction:")



@st.cache_resource
def download_pickle_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        raise Exception(f"Failed to download pickle file from {url}")

# Load the clay brick model and data
clay_model_url = 'https://github.com/MH-ML/Row-Material/raw/main/clay_model1.pkl'
clay_model = download_pickle_from_github(clay_model_url)

clay_X_supplier_url = 'https://github.com/MH-ML/Row-Material/raw/main/claybrick_supplier.pkl'
clay_X_supplier = download_pickle_from_github(clay_X_supplier_url)

scaler_url = 'https://github.com/MH-ML/Row-Material/raw/main/scaler_clay.pkl'
scaler = download_pickle_from_github(scaler_url)


def predict_dredging_match(buyer_Quality, buyer_price, buyer_availability, num_predictions=4):
    predictions = []
    suppliers = []
    predicted_classes = []
    predicted_probs = []
    for _ in range(num_predictions):
        # Get a random supplier
        supplier = clay_X_supplier.sample(1).squeeze().to_dict()
        # Make prediction
        prediction_input = np.array([[buyer_Quality, buyer_price, buyer_availability, supplier['Supplier_Quality'], supplier['Supplier_Price'], supplier['Supplier_Availability']]])
        prediction_input = scaler.transform(prediction_input)
        prediction_prob = clay_model.predict_proba(prediction_input)[0]
        
        # Apply sigmoid function to smooth out probabilities
        prediction_prob = expit(prediction_prob)
        
        prediction = 1 if prediction_prob[1] >= 0.63 else 0  # Apply threshold for classification
        predictions.append(prediction)
        suppliers.append(supplier)
        predicted_classes.append(prediction)
        predicted_probs.append(prediction_prob)
    return predictions, suppliers, predicted_classes, predicted_probs

# Streamlit UI
def main():
    st.title("Red Clay Brick Matching Between Buyer And Supplier")

    # Input fields for user
    st.sidebar.header("Buyer Details")
    buyer_details = {
        "Quality": st.sidebar.number_input("Input Grade (1 for Grade A 2 for Grade B)"),
        "Price": st.sidebar.number_input("Price"),
        "Availability": st.sidebar.number_input("Availability"),
    }

    # Prediction button
    if st.sidebar.button("Predict"):
        with st.spinner('Generating Results...'):
            predictions, suppliers, predicted_classes, predicted_probs = predict_dredging_match(
                buyer_details["Quality"], buyer_details["Price"], buyer_details["Availability"])

        st.write("---")
        st.subheader("Dashboard:")
        for i, (prediction, supplier, predicted_class, predicted_prob) in enumerate(zip(predictions, suppliers, predicted_classes, predicted_probs), start=1):
            st.write(f"Prediction {i}:")
            if predicted_class == 1:
                # If predicted class is 1, there's a match
                st.success(f"Match Score: {predicted_prob[1]:.2f}%")
                st.subheader("Buyer Details:")
                st.info("Quality: {}\nPrice: {}\nAvailability: {}".format(
                    buyer_details["Quality"], buyer_details["Price"], buyer_details["Availability"]))
                st.subheader("Supplier Details:")
                st.success("Supplier Quality: {}\nSupplier Price: {}\nSupplier Availability: {}".format(
                    supplier['Supplier_Quality'], supplier['Supplier_Price'], supplier['Supplier_Availability']))
                st.subheader("Overall Quality Match:")
                st.success(f"Overall Quality Match between Buyer and Supplier: {predicted_prob[1]:.2f}%")
            else:
                # If predicted class is 0, there's no match
                no_match_prob = 1 - predicted_prob[0]
                st.error(f"No match found. Match Score: {no_match_prob:.2f}%")
            st.write("---")

if __name__ == "__main__":
    main()