from tkinter import messagebox, simpledialog, filedialog, Text, Scrollbar, Label, Button, Tk, END
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

main = Tk()
main.title("Average Fuel Consumption")
main.geometry("1300x1200")

global filename
global train_x, test_x, train_y, test_y
global balance_data
global model
global ann_acc, svm_acc, rf_acc
global testdata, predictdata

def importdata():
    global balance_data
    balance_data = pd.read_csv(filename)
    balance_data = balance_data.abs()
    return balance_data

def splitdataset(balance_data):
    global train_x, test_x, train_y, test_y

    X = balance_data.values[:, 0:7]
    y_ = balance_data.values[:, 7]
    y_ = y_.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)  # Dense output
    Y = encoder.fit_transform(y_)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

    text.insert(END, "Dataset Length: " + str(len(X)) + "\n")
    return train_x, test_x, train_y, test_y

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n\n")

def generateModel():
    global train_x, test_x, train_y, test_y
    data = importdata()
    train_x, test_x, train_y, test_y = splitdataset(data)
    train_x = train_x.astype('float32')
    train_y = train_y.astype('float32')  # Ensure the labels are float32

    text.insert(END, "Training Length: " + str(len(train_x)) + "\n")
    text.insert(END, "Test Length: " + str(len(test_x)) + "\n")

def prediction(X_test, cls):
    y_pred = cls.predict(X_test)
    return y_pred

def ann():
    global model, ann_acc, svm_acc, rf_acc

    model = Sequential()
    model.add(Dense(200, input_shape=(7,), activation='relu', name='fc1'))
    model.add(Dense(200, activation='relu', name='fc2'))
    model.add(Dense(train_y.shape[1], activation='softmax', name='output'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print('Neural Network Model Summary:')
    print(model.summary())

    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=250)
    results = model.evaluate(test_x, test_y)
    text.insert(END, "ANN Accuracy: " + str(results[1] * 100) + "\n\n")
    ann_acc = results[1] * 100

    # Random Forest and SVM
    new_X = balance_data.values[:, 3:7]
    new_Y = balance_data.values[:, 7]
    new_train_x, new_test_x, new_train_y, new_test_y = train_test_split(new_X, new_Y, test_size=0.2)

    rfc = RandomForestClassifier(n_estimators=5, random_state=0)
    rfc.fit(new_train_x, new_train_y)
    prediction_data = prediction(new_test_x, rfc)
    rf_acc = accuracy_score(new_test_y, prediction_data) * 100
    text.insert(END, "Random Forest Accuracy: " + str(rf_acc) + "\n\n")

    svm_cls = svm.SVC(C=2.0, gamma='scale', kernel='rbf', random_state=2)
    svm_cls.fit(new_train_x, new_train_y)
    prediction_data = prediction(new_test_x, svm_cls)
    svm_acc = accuracy_score(new_test_y, prediction_data) * 100
    text.insert(END, "SVM Accuracy: " + str(svm_acc) + "\n\n")

def predictFuel():
    global testdata, predictdata
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    testdata = pd.read_csv(filename)
    testdata = testdata.values[:, 0:7]
    predictdata = np.argmax(model.predict(testdata), axis=-1)
    for i in range(len(testdata)):
        text.insert(END, str(testdata[i]) + " Average Fuel Consumption: " + str(predictdata[i]) + "\n")

def graph():
    x = list(range(len(testdata)))
    y = predictdata
    plt.plot(x, y)
    plt.xlabel('Vehicle ID')
    plt.ylabel('Fuel Consumption/10KM')
    plt.title('Average Fuel Consumption Graph')
    plt.show()

def comparisonGraph():
    height = [ann_acc, svm_acc, rf_acc]
    bars = ('ANN Accuracy', 'SVM Accuracy', 'RF Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='A Machine Learning Model for Average Fuel Consumption in Heavy Vehicles')
title.config(bg='white', fg='dodger blue', font=font, height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Heavy Vehicles Fuel Dataset", command=upload)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

modelButton = Button(main, text="Read Dataset & Generate Model", command=generateModel)
modelButton.place(x=420, y=550)
modelButton.config(font=font1)

annButton = Button(main, text="Run all 3 Algorithms", command=ann)
annButton.place(x=760, y=550)
annButton.config(font=font1)

predictButton = Button(main, text="Predict Average Fuel Consumption", command=predictFuel)
predictButton.place(x=50, y=600)
predictButton.config(font=font1)

graphButton = Button(main, text="Fuel Consumption Graph", command=graph)
graphButton.place(x=420, y=600)
graphButton.config(font=font1)

exitButton = Button(main, text="Comparison Graph All 3 Algorithms", command=comparisonGraph)
exitButton.place(x=760, y=600)
exitButton.config(font=font1)

main.config(bg='LightSkyBlue')
main.mainloop()
