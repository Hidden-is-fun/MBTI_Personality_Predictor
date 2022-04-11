import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pandas as pd
import process

accuracies = {}


def label_encode(df):
    enc = LabelEncoder()
    df['type of encoding'] = enc.fit_transform(df['type'])
    target = df['type of encoding']
    vect = CountVectorizer(stop_words='english')
    train = vect.fit_transform(df["posts"])
    # print(train.shape)

    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.4, stratify=target, random_state=42)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def Random_Forest(x_train, x_test, y_train, y_test):
    random_forest = RandomForestClassifier(n_estimators=100, random_state=1)
    random_forest.fit(x_train, y_train)

    # make predictions for test data
    Y_pred = random_forest.predict(x_test)
    predictions = [round(value) for value in Y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    accuracies['Random Forest'] = accuracy * 100.0
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


def XGBoost(x_train, x_test, y_train, y_test):
    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)

    Y_pred = xgb.predict(x_test)
    predictions = [round(value) for value in Y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    accuracies['XG Boost'] = accuracy * 100.0
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


def Gradient_Descent(x_train, x_test, y_train, y_test):
    sgd = SGDClassifier(max_iter=5, tol=None)
    sgd.fit(x_train, y_train)

    Y_pred = sgd.predict(x_test)
    predictions = [round(value) for value in Y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    accuracies['Gradient Descent'] = accuracy * 100.0
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


def Logistic_Regression(x_train, x_test, y_train, y_test):
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    Y_pred = logreg.predict(x_test)
    predictions = [round(value) for value in Y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    accuracies['Logistic Regression'] = accuracy * 100.0
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


def KNN(x_train, x_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=2)  # n_neighbors means k
    knn.fit(x_train, y_train)

    Y_pred = knn.predict(x_test)
    predictions = [round(value) for value in Y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    accuracies['KNN'] = accuracy * 100.0
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # try to find best k value
    scoreList = []
    for i in range(1, 20):
        knn2 = KNeighborsClassifier(n_neighbors=i)  # n_neighbors means k
        knn2.fit(x_train, y_train)
        scoreList.append(knn2.score(x_test, y_test))

    acc = max(scoreList) * 100

    print("Maximum KNN Score is {:.2f}%".format(acc))


def RF_4axis(X, personality_type, list_personality):
    for l in range(len(personality_type)):
        Y = list_personality[:, l]

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

        # fit model on training data
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # make predictions for test data
        y_pred = model.predict(X_test)

        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)

        print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))


def XGB_4axis(X, personality_type, list_personality):
    for l in range(len(personality_type)):
        Y = list_personality[:, l]

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        param = {}
        param['n_estimators'] = 200
        param['max_depth'] = 2
        param['nthread'] = 1024
        param['learning_rate'] = 0.2

        # fit model on training data
        model = XGBClassifier(**param)
        model.fit(X_train, y_train)

        # make predictions for test data
        y_pred = model.predict_proba(X_test)
        print(y_pred)
        # predictions = [round(value) for value in y_pred]
        # model.get_booster().save_model(f'models/{personality_type[l][:2]}.model')
        joblib.dump(model, f'models/{personality_type[l][:2]}.model')
        # evaluate predictions
        # accuracy = accuracy_score(y_test, predictions)

        # print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))
