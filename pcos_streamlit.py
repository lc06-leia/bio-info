import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
#Import Modules for machine learning and deep learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier




data = pd.read_csv(r'PCOS_data_without_infertility.csv')

st.set_page_config(page_title = 'PCOS Diagnosis Prediction', layout = 'wide')

st.write("""
# PCOS Diagnosis Prediction
""")


st.sidebar.header('Models & Hyperparameters')
model = st.sidebar.selectbox(
    'Select the Model: ',
    options = np.array(['Logistic Regression', 'Decision Tree', 'Neural Network', 'No Model'])
)
background_info = st.sidebar.selectbox(
    'Show Background Information & Dataset: ',
    options = np.array(['Yes', 'No'])
)

st.sidebar.write("""
#### Data Source:

Kottarathil, Prasoon. “Polycystic Ovary Syndrome (PCOS).” Kaggle, 2020, Accessed 2024. 
""")


data = data.drop(['Unnamed: 41'], axis=1)
columns_objects = ['AMH(ng/mL)', 'II    beta-HCG(mIU/mL)', 'Sl. No', 'Patient File No.']


data = data.drop(columns_objects, axis = 1)
data = data.dropna()

data_original = data.copy()

if background_info == 'Yes':
    #Background information

    st.write("""
    ### What is Polycystic Ovary Syndrome (PCOS)?
    
    Polycystic Ovary Syndrome (PCOS) is a common condition in women which affects hormones. 
    PCOS can cause irregular periods, acne, and excess hair growth (on arms, chest, abdomen, and face). Concerningly, PCOS can also cause infertility
    as well as may increase the risk for diabetes and high blood pressure.
    
    The ovary produces excess androgen (a hormone) causing unpredictable periods and ovulation, this can result in follicle cysts 
    on the ovaries which can help diagnose the condition. Androgen prevents ovaries from releasing eggs causing irregular periods.
    Additionally, androgen increases acne and hair growth in women. 
    
    While people are mostly diagnosed in their twenties and thirties, PCOS can occur at any point after puberty. 
    
    Source: Cleveland Clinic [link](https://my.clevelandclinic.org/health/diseases/8316-polycystic-ovary-syndrome-pcos)
    """)

    #Show Dataset

    st.write("""
    ### PCOS Data """)
    st.dataframe(data_original)

    #Create Correlation Matrix

    st.write("""
    ### Correlation Matrix
    This correlation matrix shows the correlation between seven randomly chosen features and PCOS status. 
    Rerun the program to see other features.
    """)

    data_no_pcos = data.drop(['PCOS (Y/N)'], axis = 1)
    column_names = data_no_pcos.columns
    random_columns = np.random.choice(column_names, 7)
    random_columns = np.append(random_columns, 'PCOS (Y/N)').tolist()

    data_random_columns = data[random_columns]

    corr_matrix = data_random_columns.corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, ax = ax, annot = True, cmap = 'Blues')
    st.write(fig)

    # Describe the Data

    st.write("""
    
    ### Description of the Data
    The dataframe below summarizes the values found in the PCOS Data including the quartile values, the mean, the standard deviation, 
    and the count.

    """)

    data_describe = data.describe()
    st.dataframe(data_describe)



else:
    st.write("Click 'Yes' on Background Information to see Dataset")

y = data[['PCOS (Y/N)']]
x = data.drop(['PCOS (Y/N)'], axis = 1)

features_num = [' Age (yrs)', 'Weight (Kg)', 'Height(Cm) ', 'Pulse rate(bpm) ', 'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)',
    'Cycle length(days)', 'Marraige Status (Yrs)', 'No. of aborptions', '  I   beta-HCG(mIU/mL)', 'FSH(mIU/mL)',
    'LH(mIU/mL)', 'Hip(inch)', 'Waist(inch)', 'TSH (mIU/L)', 'PRL(ng/mL)',
    'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Follicle No. (L)',
    'Follicle No. (R)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)',
    'Endometrium (mm)']

features_cat = ['Blood Group', 'Pregnant(Y/N)', 'Weight gain(Y/N)','hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)',
    'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)']

transformer_num = make_pipeline(SimpleImputer(strategy="constant"), StandardScaler())

transformer_cat = make_pipeline(SimpleImputer(strategy="constant"),OneHotEncoder())

preprocessor = make_column_transformer((transformer_num, features_num),(transformer_cat, features_cat))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 42)

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)


if model == 'Neural Network':

    st.write("""
    ## Neural Network Results
    """)

    num_epochs = st.sidebar.slider('Number of Epochs', min_value = 1, max_value = 1000, value = 50)
    num_batches = st.sidebar.slider('Number of Batches', min_value = 1, max_value = 1000, value = 10)
    first_layer_density = st.sidebar.slider('First Layer Density', min_value=1, max_value=500, value = 60)
    second_layer_density = st.sidebar.slider('Second Layer Density', min_value=1, max_value=500, value = 30)


    input_shape = [x_train.shape[1]]

    neural_model = keras.Sequential([
        layers.BatchNormalization(input_shape=input_shape),
        layers.Dense(first_layer_density, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(second_layer_density, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    neural_model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['binary_accuracy']
    )

    early_stopping = keras.callbacks.EarlyStopping(
        patience =5,
        min_delta = 0.001,
        restore_best_weights = True
    )

    neural_model_final = neural_model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = num_batches,
                                    epochs = num_epochs, callbacks = [early_stopping])

    history_df = pd.DataFrame(neural_model_final.history)
    fig1 = history_df.loc[:, ['loss', 'val_loss']]
    fig2 = history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']]
    st.write("""
        #### Cross-entropy
    """)
    st.line_chart(fig1)
    st.write("""
        #### Accuracy
    """)
    st.line_chart(fig2)

elif model == 'Logistic Regression':
    st.write("""
    ## Logistic Regression Results
    """)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Showing the accuracy

    accuracy = model.score(x_test, y_test)

    st.write("""
    ### Model Accuracy is:""")
    st.write(accuracy)

    # Showing the confusion matrix

    st.write("""
    ### Confusion Matrix
    
    The confusion matrix shows the number of instances of true positives, false positives, true negatives, and false 
    negatives.
    
    """)
    y_pred_test = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred_test)

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])

    fig, ax = plt.subplots()

    sns.heatmap(cm_matrix, ax = ax, annot=True, fmt='d', cmap='YlGnBu')
    st.write(fig)

    # Showing the Classification Report

    st.write("""
    ### Classification Report

    #### Precision
    The percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. 
    The ratio of true positives to the sum of true and false positives. 

    #### Recall
    The percentage of correctly predicted positive outcomes out of all the actual positive outcomes. 
    Can be given as the ratio of true positives to the sum of true positives and false negativies. 
    Recall is also called sensitivity. TP/ (TP + FN)

    #### f1-score
    The weighted harmonic mean of precision and recall. The best score would be 1.0 and the 0.0. 
    Always lower than accuracy measures.

    #### Support
    The actual number of occurances of the class in our dataset

    """)

    report = classification_report(y_test, y_pred_test, output_dict = True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)


elif model == 'Decision Tree':
    st.write("""
    ## Decision Tree Results
    """)
    model =  DecisionTreeClassifier(criterion = 'gini', max_depth = 2, random_state = 1)
    model.fit(x_train, y_train)

    # Showing the accuracy

    accuracy = model.score(x_test, y_test)

    st.write("""
    ### Model Accuracy is:""")
    st.write(accuracy)

    # Showing the confusion matrix

    st.write("""
    ### Confusion Matrix

    The confusion matrix shows the number of instances of true positives, false positives, true negatives, and false 
    negatives.

    """)
    y_pred_test = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred_test)

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])

    fig, ax = plt.subplots()

    sns.heatmap(cm_matrix, ax=ax, annot=True, fmt='d', cmap='YlGnBu')
    st.write(fig)

    # Showing the Classification Report

    st.write("""
    ### Classification Report

    #### Precision
    The percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. 
    The ratio of true positives to the sum of true and false positives. 

    #### Recall
    The percentage of correctly predicted positive outcomes out of all the actual positive outcomes. 
    Can be given as the ratio of true positives to the sum of true positives and false negativies. 
    Recall is also called sensitivity. TP/ (TP + FN)

    #### f1-score
    The weighted harmonic mean of precision and recall. The best score would be 1.0 and the 0.0. 
    Always lower than accuracy measures.

    #### Support
    The actual number of occurances of the class in our dataset

    """)

    report = classification_report(y_test, y_pred_test, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

else:
    st.write('Please choose a model to see results.')
