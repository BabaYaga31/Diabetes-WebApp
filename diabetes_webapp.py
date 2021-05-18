import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
# ConfusionMatrixDisplay
# plot_roc_curve
# plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.write(""" 
# Diabetes Detection
Detect if someone has diabetes using machine learning and python !
""")

image = Image.open(
    'diabetes_logo.png')
st.image(image, caption='ML', use_column_width=True)

# Getting the data

df = pd.read_csv('new_diabetes.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('Unnamed: 0.1', axis=1, inplace=True)
# Subheader
st.subheader('Data Information: ')
st.dataframe(df)

# Statistics on data
st.write(df.describe())

# Show the data as a chart
chart = st.bar_chart(df)

# Splitting data into dependent and independent
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

# Split into test and training
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# Getting the feature input from the user


def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 13, 3)
    glucose = st.sidebar.slider('glucose', 44, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 38, 106, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 7, 63, 23)
    insulin = st.sidebar.slider('insulin', 14.0, 318.0, 30.0)
    BMI = st.sidebar.slider('BMI', 18.0, 50.0, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.00, 0.3725)
    age = st.sidebar.slider('age', 21, 66, 29)

    user_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'BMI': BMI,
        'DPF': DPF,
        'age': age
    }

    features = pd.DataFrame(user_data, index=[0])
    return features


# Store user input into variable
user_input = get_user_input()

st.subheader('User Input:')
st.write(user_input)

# Create and train the model
RandomForesClassifier = RandomForestClassifier(
    criterion='entropy', n_estimators=200, random_state=0)
RandomForesClassifier.fit(X_train, Y_train)

y_pred = RandomForesClassifier.predict(X_test)
y_prob = RandomForesClassifier.predict_proba(X_test)[:, 1]
cm = confusion_matrix(Y_test, y_pred)

# Show the metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForesClassifier.predict(X_test)) * 100) + '%')

# Prediction
prediction = RandomForesClassifier.predict(user_input)

# Display result
st.subheader('Classification: ')
st.write(prediction)


# # Visualizing Confusion Matrix
# plt.figure(figsize = (6, 6))
# sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15},
#             yticklabels = ['Healthy', 'Diabetic'], xticklabels = ['Predicted Healthy', 'Predicted Diabetic'])
# plt.yticks(rotation = 0)
# plt.show()
# st.pyplot()

# # Roc AUC Curve
# false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_prob)
# roc_auc = auc(false_positive_rate, true_positive_rate)

# sns.set_theme(style = 'white')
# plt.figure(figsize = (6, 6))
# plt.plot(false_positive_rate,true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
# plt.axis('tight')
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.title('ROC AUC Curve')
# plt.legend()
# plt.show()
# st.pyplot()

# #Precision Recall Curve
# average_precision = average_precision_score(Y_test, y_prob)
# disp = plot_precision_recall_curve(RandomForesClassifier, X_test, Y_test)
# plt.title('Precision-Recall Curve')
# plt.show()
# st.pyplot()
