import streamlit as st
import numpy as np
import joblib

@st.cache_resource
def load_artifacts():
 model=joblib.load("models/rf_final.pkl")
 encoder=joblib.load("models/target_encoder.pkl")
 onehot=joblib.load("models/onehot_encoder.pkl")
 scaler=joblib.load("models/scale.pkl")
 imputer=joblib.load("models/imputer (1).pkl")
 return model, encoder, onehot, scaler, imputer

model, encoder, onehot, scaler,imputer = load_artifacts()

st.title("Forest Cover Type Prediction App")
st.write("Enter the values for the following featutes")

elevation=st.number_input("Elevation",min_value=1863,
                          max_value=3849,value=2000)
dist_road=st.number_input("Horizontal Distance To Roadways",min_value=0,
                          max_value=7117,value=500)
dist_fire=st.number_input("Horizontal Distance To Fire Points",min_value=0,
                          max_value=7173,value=600)
dist_hydro=st.number_input("Horizontal Distance To Hydrology",min_value=0,
                           max_value=1343,value=300)
vertical_hydro=st.number_input("Vertical Distance To Hydrology",min_value=-146,
                               max_value=554,value=50)

elevation_hydrology = elevation - vertical_hydro
if vertical_hydro != 0:
    hydrology = dist_hydro / vertical_hydro + 1
else:
    hydrology = dist_hydro

aspect=st.slider("Aspect",0,360,90)
slope=st.slider("slope",0,61,10)
hillshade_9am=st.slider("Hillshade_9am",0,255,150)
hillshade_noon=st.slider("Hillshade_noon",0,255,200)
hillshade_3pm=st.slider("Hillshade_3pm",0,255,100)

wilderness=st.selectbox("Wilderness",[1,2,3,4])
soil_type=st.selectbox("Soil Type",list(range(1,41)))
if st.button("Predict"):
    with st.spinner("Predicting..."):
     num_features=np.array([[elevation,dist_road,dist_fire,dist_hydro,
                        vertical_hydro,elevation_hydrology,hydrology,
                        aspect,slope,hillshade_9am,hillshade_noon,hillshade_3pm]])
    cat_features=np.array([[wilderness,soil_type]])
    cat_encoded = onehot.transform(cat_features)
    if hasattr(cat_encoded, "toarray"):
      cat_encoded = cat_encoded.toarray()
    final_features=np.hstack((num_features,cat_encoded))
    final_features=imputer.transform(final_features)
    
    final_scaled=scaler.transform(final_features) 
    prediction=model.predict(final_scaled)
    final_output=encoder.inverse_transform(prediction)    
    
    st.success(f" Predicted Forest Cover Type:{final_output[0]}")