import numpy as np

def corr_graph(data, features):
    
    import matplotlib.pyplot as plt
    
    data = data[features]
    corr = data.corr().style.background_gradient(cmap='coolwarm', axis=None)
    # corr = corr.style.background_gradient(cmap='coolwarm', axis=None)
    # corr = corr.style.background_gradient(cmap='coolwarm')
    # corr = corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    plt.matshow(corr)
    plt.show()

def class_balance(classes, y):
    from yellowbrick.target import ClassBalance
    visualizer = ClassBalance(labels=classes)
    visualizer.fit(y)
    visualizer.poof()


def rank_2d(features, algorithm, X, y ):
    from yellowbrick.features import Rank2D

    # Instantiate the visualizer with the Covariance ranking algorithm
    visualizer = Rank2D(features=features, algorithm=algorithm)

    visualizer.fit(X, y)                # Fit the data to the visualizer
    visualizer.transform(X)             # Transform the data
    visualizer.poof()                   # Draw/show/poof the data
def feature_selection(model, features,X, y ):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from yellowbrick.features import RFECV

    # df = load_data('credit')

    
    # features = [col for col in data.columns if col != target]

    # X = data[features]
    # y = data[target]

    cv = StratifiedKFold(5)
    oz = RFECV(model, cv=cv, scoring='f1_weighted')

    oz.fit(X, y)
    oz.poof()
    
def pearson_correlation(classes, fetures, X, Y):
    from sklearn import datasets
    from yellowbrick.target import FeatureCorrelation

    # Load the regression data set
    # data = datasets.load_diabetes()
    # X, y = data['data'], data['target']
    # feature_names = np.array(data['feature_names'])

    visualizer = FeatureCorrelation(labels=fetures)
    visualizer.fit(X, Y)
    visualizer.poof()
    
def mutual_info_classification(classes, feature_names, X, y):
    from sklearn import datasets
    from yellowbrick.target import FeatureCorrelation

    # Load the regression data set
    
    visualizer = FeatureCorrelation(method='mutual_info-classification',
                                    feature_names=feature_names, sort=True)
    visualizer.fit(X, y, random_state=0)
    visualizer.poof()
    
def mutual_info_regress(classes, feature_names, X, y):
    from sklearn import datasets
    from yellowbrick.target import FeatureCorrelation

    # Load the regression data set


    

    discrete_features = [False for _ in range(len(feature_names))]
    discrete_features[1] = True

    visualizer = FeatureCorrelation(method='mutual_info-regression',
                        labels=feature_names)
    visualizer.fit(X, y, discrete_features=discrete_features, random_state=0)
    visualizer.poof()

def validation_curve(model, X, y):
    from yellowbrick.model_selection import ValidationCurve
    from sklearn.model_selection import StratifiedKFold
    # Create the validation curve visualizer
    cv = StratifiedKFold(12)
    # param_range = np.linspace(30.00, 300.00, num=50.00, dtype=np.float64)
    param_range = np.logspace(30, 300, num=100, dtype = np.int32)

    viz = ValidationCurve(
        model, param_name="n_estimators", param_range=param_range,
        logx=True, cv=cv, scoring="f1_weighted", n_jobs=8,
    )

    viz.fit(X, y)
    viz.poof()

def class_predict_error(model, classes, X_train, Y_train, X_test, Y_test):
    from yellowbrick.classifier import ClassPredictionError

    # Instantiate the classification model and visualizer
    visualizer = ClassPredictionError(
        RandomForestClassifier(), classes=classes
    )

    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)

    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # Draw visualization
    g = visualizer.poof()
    
def classification_report(model, classes, X_train, Y_train, X_test, Y_test):
    
    from yellowbrick.classifier import ClassificationReport

    # Instantiate the classification model and visualizer
    
    visualizer = ClassificationReport(model, classes=classes, support=True)

    visualizer.fit(X_train, Y_train)  # Fit the visualizer and the model
    visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
    g = visualizer.poof()             # Draw/show/poof the data
    
def ROC_AUC(model, classes, X_train, Y_train, X_test, Y_test):
    from yellowbrick.classifier import ROCAUC

    # Instantiate the visualizer with the classification model
    visualizer = ROCAUC(model, classes=classes)

    visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
    g = visualizer.poof() 

def precision_recall(model, classes, X_train, Y_train, X_test, Y_test):
    from yellowbrick.classifier import PrecisionRecallCurve

# Load the dataset and split into train/test splits

    # Create the visualizer, fit, score, and poof it
    viz = PrecisionRecallCurve(model)
    viz.fit(X_train, Y_train)
    viz.score(X_test, Y_test)
    viz.poof()   
    
def precision_recall_f1(model, classes, X_train, Y_train, X_test, Y_test):
    from yellowbrick.classifier import PrecisionRecallCurve
    viz = PrecisionRecallCurve(
    model, per_class=True, iso_f1_curves=True,
    fill_area=False, micro=False
    )
    viz.fit(X_train, Y_train)
    viz.score(X_test, Y_test)
    viz.poof()
    
def confusion_matrix(model, classes, X_train, Y_train, X_test, Y_test):
    from yellowbrick.classifier import ConfusionMatrix
    iris_cm = ConfusionMatrix(
    model, classes=classes,
    label_encoder={0: classes[0], 1: classes[1]}
    )

    iris_cm.fit(X_train, Y_train)
    iris_cm.score(X_test, Y_test)

    iris_cm.poof()
    
def discrimination_thersold(model, classes, X_train, Y_train, X_test, Y_test):
    from yellowbrick.classifier import DiscriminationThreshold

    # Instantiate the classification model and visualizer
    
    viz = DiscriminationThreshold(model)

    # visualizer.fit(X, y)  # Fit the training data to the visualizer
    # visualizer.poof()     # Draw/show/poof the data
    viz.fit(X_train, Y_train)
    # viz.score(X_test, Y_test)
    viz.poof()

def learning_curve(model, X, y):
    # from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import RepeatedStratifiedKFold

    from yellowbrick.model_selection import LearningCurve

    # Create the learning curve visualizer
    # cv = StratifiedKFold(12)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    sizes = np.linspace(0.3, 1.0, 10)
   
    viz = LearningCurve(
        model, cv=cv, train_sizes=sizes,
        scoring='neg_log_loss', n_jobs=4
    )

    # Fit and poof the visualizer
    viz.fit(X, y)
    viz.poof()
    
    
def decision_boundary(model,features, classes, X_train, Y_train, X_test, Y_test):
    from yellowbrick.contrib.classifier import DecisionViz
    features = ['name_sim', 'add_sim']
    viz = DecisionViz(model, title="random forest", features=features, classes=classes)
    viz.fit(X_train, Y_train)
    viz.draw(X_test, Y_test)
    viz.poof()
