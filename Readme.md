**Dataset Description**
        
    Dataset Link -- http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
    Download File -- wdbc.data
    Change wdbc.data to data.csv
    
    wdbc.data:
         Number of instances: 569 
         Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)
         Class distribution: 357 benign, 212 malignant
    
**Classifiers Used**  
    
    1. Logistic Regression
    2. SVM - Linear Kernel
    3. SVM - RBF Kernel
    4. MLPClassifier
    5. KNN
    6. Random Forest Classifier
    7. Gaussian Naive Bayes Classifier
    8. Neural Network(2 hidden layers, 100 units in first layer, 75 units in second layer, 40 epochs)
    
**Results**
        
        Classifier                  |      Accuracy
     ----------------------------------------------
    1. Logistic Regression              -- 99.12 %
    2. SVM - Linear Kernel              -- 99.12 %
    3. SVM - RBF Kernel                 -- 99.12 %
    4. MLPClassifier                    -- 99.12 %
    5. KNN                              -- 95.61 %
    6. Random Forest Classifier         -- 95.61 %
    7. Gaussian Naive Bayes Classifier  -- 92.98 % 
    8. Neural Network                   -- 100 %