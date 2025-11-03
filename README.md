# Fruit & Vegetable Leaf Disease Detection System (Streamlit + TensorFlow)

🔗 **Live Demo:** 
                  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fruit-and-leaf-disease-detection-system-b5eerpcj6acy5qk5bcqlev.streamlit.app/)

Interactive Streamlit app that classifies fruit/vegetable leaf diseases from images using a TensorFlow/Keras CNN, and suggests pesticide guidance per disease class.

## Overview
- Upload a leaf image (JPG/PNG) and get the predicted disease class among 28 categories across Apple, Corn, Grape, Orange, Potato, Sugarcane, and Tomato.
- Backend: TensorFlow model loaded once for fast inference; preprocessing uses PIL and resizing to 128×128.
- Frontend: Streamlit with simple navigation (Home, Leaf Disease Recognition, About).

## How to run locally
1. Python 3.10+
2. Install dependencies:
   pip install -r requirements.txt
3. Ensure the model file `f&v_leaf_detection_model.keras` is in the project root and `assets/home_plant.jpg` exists.
4. Start app:
    Option A : Double‑click `app.bat` to launch the app in your default browser.
    Option B: `streamlit run app.py` in the terminal.


## Files
- `app.py` — Streamlit UI and inference.
- `app.bat` — Launch the app in the default browser.
- `f&v_leaf_detection_model.keras` — trained Keras model.
- `assets/` — static image(s).
- `notebook/` — training notebook.
- `test/` — sample test images.


## Notes
- CLASS_NAMES order must match the model’s training labels.
- Normalize images if the model expects [0,1] input.
- Ongoing development; stability and performance are not yet final.
