import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(page_title="IS PROJECT by GULLANUT", layout="wide")

menu = st.sidebar.radio("Menu", ["Machine Learning", "Neural Network", "Predicting Video Game", "Predicting Series"])

# --- หน้า 1: Machine Learning ---
if menu == "Machine Learning":
    st.title("Machine Learning - Decision Tree and Random Forest")
    st.markdown("""
    ### **1. Load Data from CSV from Kaggle**
    - Load the dataset from `vgsales.csv`, which contains video game sales data.
    - Use `pd.read_csv` to read the data into the `data` variable.

    ### **2. Handle Missing Values**
    - Fill missing values in the `Year` column with the median value.
    - Fill missing values in the `Publisher` column with `"Unknown"`.

    ### **3. Convert Categorical Data to Numeric**
    - Use `pd.get_dummies` to convert categorical columns (`Platform`, `Genre`, `Publisher`) into numerical values using **One-Hot Encoding**.

    ### **4. Prepare Data for Model Training**
    - Select features for training (`X`) and target labels (`y`).
    - Remove irrelevant columns (`Rank`, `Name`, `Global_Sales`) from `X`.
    - Convert `Global_Sales` into a categorical target (`High` or `Low` sales).

    ### **5. Train Decision Tree and Random Forest Models**
    - Train **Decision Tree** and **Random Forest** models using training data (`X_train` and `y_train`).
    - These models will predict video game sales.

    ### **6. Save the Models and Features**
    - Save the trained models (both Decision Tree and Random Forest) as `.pkl` files.
    - Store the feature names used for training to ensure correct input processing later.

    ### **7. Build an Interface with Streamlit**
    - Use **Streamlit** to create an interactive interface.
    - Allow users to select **Platform, Genre, and Release Year**.

    ### **8. Process User Input**
    - Convert user input into a format compatible with the trained model.
    - Map user-selected values (Platform, Genre, Year) to match model features.

    ### **9. Find Matching Games**
    - Search for games that match the user’s selection in the dataset.
    - Display a list of recommended games.

    ### **10. Predict Game Sales**
    - When the user clicks **"Predict"**, the saved models are loaded.
    - Both **Decision Tree** and **Random Forest** models predict whether the game has **High or Low Sales**.

    ### **11. Recommend Games Based on Predictions**
    - Display up to **5 games** matching the user’s criteria.
    - Show a message if no games match the selected criteria.
    """)

# --- หน้า 2: Neural Network ---
elif menu == "Neural Network":
    st.title("Neural Network-based Series Title Prediction")
    st.markdown("""
### 1. **Preparing Data and Training the Model**
The code begins by training a model to predict the name of a series based on the given **Genre** and **Release Year**.
- It uses `MultiLabelBinarizer` to convert the **Genre** from a string format (e.g., "Drama, Comedy") into numerical values, which can be used to train the model.
- `LabelEncoder` is used to convert the **Series Name** into numeric values so that the model can learn from them.
- The model is a **Neural Network** (using `Sequential` and `Dense` layers) with 3 layers:
  - First layer: 64 units with `ReLU` activation.
  - Second layer: 32 units with `ReLU` activation.
  - Final layer: Uses `softmax` activation to produce probabilities for each series.
- After training, the model is saved in the file `series_model.h5`.

### 2. **Loading the Trained Model**
When the page for predicting series is opened, the system attempts to load the previously trained model (`series_model.h5`) along with the related data for encoding **Genre** and **Series Name** (such as `LabelEncoder` and `MultiLabelBinarizer`).
- If the necessary files are not found, the system will train the model again.

### 3. **Predicting the Series Name**
When the user selects a **Genre** and **Release Year**, these inputs are used to predict the series name.
- `MultiLabelBinarizer` converts the selected **Genre** into numerical values, which are then combined with the **Release Year** to form the input for the model.
- The trained model is used to make a prediction by calculating the probabilities of each series name based on the given inputs.
- The predicted index is then transformed back into a series name.

### 4. **Validating the Prediction**
Before displaying the result, the predicted series must be checked against the database to ensure it matches the selected **Genre** and **Release Year**.
- If the predicted series is found in the list of valid series (`valid_series`) that match the selected **Genre** and **Release Year**, it will be displayed.
- If the predicted series is not found in the valid series, no result will be shown.

### 5. **Building the UI with Streamlit**
In the UI (user interface), the user can select a **Genre** and **Release Year** from dropdown boxes (`selectbox`).
- When the user clicks the **Predict Series Name** button, the `predict_series` function is called to generate the predicted series name.
- If the prediction is valid (matching the database), the system will display the message: `Predicted Successful: **Predicted Series Name**`.

### 6. **Results**
- The system first shows the valid series matching the selected **Genre** and **Release Year** before making a prediction.
- If the predicted series matches the database, the system will display the name of the predicted series.
- If no match is found, the system will not display any series name.
""")


# --- หน้า 3: Predicting Video Game ---
elif menu == "Predicting Video Game":
    st.title("Recommended games by sales")
    file_path = "vgsales.csv"
    data = pd.read_csv(file_path)

    # เติมค่าที่หายไป
    data['Year'].fillna(data['Year'].median(), inplace=True)
    data['Publisher'].fillna('Unknown', inplace=True)

    # แปลงข้อมูลหมวดหมู่เป็นตัวเลข
    data = pd.get_dummies(data, columns=['Platform', 'Genre', 'Publisher'])

    # เก็บข้อมูลชื่อเกมเพื่อนำมาแนะนำ
    game_data = data[['Name', 'Year'] + list(data.columns[3:])]

    # เลือกฟีเจอร์และเป้าหมาย
    X = data.drop(['Rank', 'Name', 'Global_Sales'], axis=1)
    y = data['Global_Sales'].apply(lambda x: 1 if x > 1 else 0)  # แบ่งกลุ่มยอดขายสูงและต่ำ

    # บันทึกรายชื่อฟีเจอร์เพื่อใช้ในภายหลัง
    feature_names = list(X.columns)
    joblib.dump(feature_names, 'feature_names.pkl')

    # แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # สร้างและฝึก Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    # สร้างและฝึก Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # บันทึกโมเดลลงในไฟล์ .pkl
    joblib.dump(dt_model, 'decision_tree_model.pkl')  # บันทึกโมเดล Decision Tree
    joblib.dump(rf_model, 'random_forest_model.pkl')  # บันทึกโมเดล Random Forest

    # --- ส่วนของ Streamlit --- 
    st.title("Recommended games by sales")

    # เลือกแพลตฟอร์ม, ประเภทเกม และปีที่วางจำหน่าย
    platform = st.selectbox("Select platform", ['Wii', 'NES', 'GB', 'PS4', 'X360'])
    genre = st.selectbox("Select game type", ['Sports', 'Racing', 'Role-Playing', 'Shooter', 'Action'])
    year = st.slider("Select release year", 1980, 2020, 2000)

    # แปลงข้อมูลอินพุต
    user_input = pd.DataFrame({
        'Year': [year],
        'Platform_Wii': [1 if platform == 'Wii' else 0],
        'Platform_NES': [1 if platform == 'NES' else 0],
        'Platform_GB': [1 if platform == 'GB' else 0],
        'Platform_PS4': [1 if platform == 'PS4' else 0],
        'Platform_X360': [1 if platform == 'X360' else 0],
        'Genre_Sports': [1 if genre == 'Sports' else 0],
        'Genre_Racing': [1 if genre == 'Racing' else 0],
        'Genre_Role-Playing': [1 if genre == 'Role-Playing' else 0],
        'Genre_Shooter': [1 if genre == 'Shooter' else 0],
        'Genre_Action': [1 if genre == 'Action' else 0],
    })

    # โหลดฟีเจอร์จากไฟล์ที่บันทึกไว้
    feature_names = joblib.load('feature_names.pkl')

    # ทำให้ข้อมูลอินพุตตรงกับฟีเจอร์ที่ใช้ในโมเดล
    for feature in feature_names:
        if feature not in user_input.columns:
            user_input[feature] = 0

    user_input = user_input[feature_names]

    # ค้นหาเกมที่ตรงกับตัวเลือกของผู้ใช้
    recommended_games = game_data[ 
        (game_data['Year'] == year) & 
        (game_data[f'Platform_{platform}'] == 1) & 
        (game_data[f'Genre_{genre}'] == 1)
    ]['Name'].tolist()

    # ทำนาย
    if st.button("Predict"):
        with st.spinner('Predicting...'):
            # โหลดโมเดลที่บันทึกไว้
            dt_model = joblib.load('decision_tree_model.pkl')
            rf_model = joblib.load('random_forest_model.pkl')

            # ทำนายยอดขาย
            dt_prediction = dt_model.predict(user_input)
            rf_prediction = rf_model.predict(user_input)

            st.write(f"Decision Tree: {'High sales' if dt_prediction[0] == 1 else 'Low sales'}")
            st.write(f"Random Forest: {'High sales' if rf_prediction[0] == 1 else 'Low sales'}")

            # แนะนำเกม
            if recommended_games:
                st.write("Recommended games matching your selection:")
                for game in recommended_games[:5]:  # แสดงสูงสุด 5 เกม
                    st.write(f"- {game}")
            else:
                st.write("No games found matching your selection")

# --- หน้า 4: Predicting Series ---
elif menu == "Predicting Series":
    st.title('Predicting Title of Series')

    # --- ฟังก์ชันฝึกโมเดล ---
    def train_model(df):
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(df['Genre'].str.split(', '))

        X = df[['Release Year']] 
        X = np.hstack((X, genre_encoded))

        y = df['Series Name']
        le_series = LabelEncoder()
        y_encoded = le_series.fit_transform(y)  

        X = X.astype(np.float32)
        y_encoded = y_encoded.astype(np.int32)

        model = Sequential()
        model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X, y_encoded, epochs=50, batch_size=10)

        model.save('series_model.h5')

        return le_series, mlb, model

    # --- ฟังก์ชันโหลดโมเดล ---
    def load_trained_model():
        model = load_model('series_model.h5')
        return model

    # --- ฟังก์ชันทำนายชื่อซีรีส์ พร้อมตรวจสอบผลลัพธ์ ---
    def predict_series(model, le_series, mlb, genre, release_year, df):
        genre_encoded = mlb.transform([genre.split(', ')])[0]  
        features = np.array([[release_year] + genre_encoded.tolist()])

        # ทำนายชื่อซีรีส์
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_series = le_series.inverse_transform([predicted_index])[0]

        # ตรวจสอบว่ามีข้อมูลที่ตรงกับ Genre และ Release Year หรือไม่
        valid_series = df[(df['Release Year'] == release_year) & (df['Genre'].str.contains(genre, regex=False))]

        # แสดงข้อมูลที่ตรงกับ Genre และ Release Year
        st.write(f"Valid series matching genre '{genre}' and release year {release_year}: {valid_series['Series Name'].values}")

        # หากไม่พบข้อมูลที่ตรงกันกับ Genre และ Release Year ให้แสดงผลว่า "ไม่มี"
        if predicted_series not in valid_series['Series Name'].values:
            return None  # ถ้าไม่พบจะคืนค่า None

        return predicted_series

    # --- สร้าง UI ด้วย Streamlit ---
    st.title('Predicting Series')

    df = pd.read_csv('series.csv')

    # โหลดโมเดลที่ฝึกแล้วหากมี
    try:
        model = load_trained_model()
        st.write("✅ The model is ready to use.")
        le_series = joblib.load('le_series.pkl')
        mlb = joblib.load('mlb.pkl')
    except:
        st.write("⏳ Training the model...")

        le_series, mlb, model = train_model(df)  
        model = load_trained_model()

        joblib.dump(le_series, 'le_series.pkl')
        joblib.dump(mlb, 'mlb.pkl')

    # ให้ผู้ใช้เลือกข้อมูลจากข้อมูลที่มี
    genre_input = st.selectbox('🎭 Select Genre:', mlb.classes_)  
    release_year_input = st.selectbox('📅 Select Release Year:', df['Release Year'].unique())

    if st.button('🔍 Predict Series Name'):
        predicted_series = predict_series(model, le_series, mlb, genre_input, release_year_input, df)
    
        if predicted_series:
            st.write(f'✅ Predicted Successful: **{predicted_series}**')