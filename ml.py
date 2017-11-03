import numpy as np

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier

def getArray(label, csvName):   # csvList =[]

    #for fileName in csvList:  
        data = np.loadtxt(csvName , delimiter=',', usecols=range(7,19), skiprows=1 )
        
        # Drop Row if one of quality indexes are below 10.
        quality_indexes = [1,3,5,7,9,11]           # range(1, len(data[0]), 2)
        for quality_index in quality_indexes:
            data = data[ np.logical_not( data[:, quality_index] < 10 ) ]
            #print data.shape
        
        # Keep only O1.....
        data = data[:,[0,2,4,6,8,10]]
    
        num_rows, num_cols = data.shape
        label_column = np.full((num_rows, 1), label)  #.0, dtype=np.float32
        #print data.shape    print label_column.shape 

        data = np.hstack(( data , label_column ))
        # print data[0]

        #print csvName + "  =>  " + str(label)
        #print data.shape

        return data

    #return output


def main():
    '''
    blue_csv = ['raw_hansika_blue.csv', 'raw_heshan_blue.csv', 'raw_dinuka_blue.csv', 'raw_nadun_blue.csv', 'raw_ravindu_blue.csv']
    green_csv = ['raw_hansika_green.csv', 'raw_heshan_green.csv', 'raw_dinuka_green.csv', 'raw_nadun_green.csv', 'raw_ravindu_green.csv']
    red_csv = ['raw_hansika_red.csv', 'raw_heshan_red.csv', 'raw_dinuka_red.csv', 'raw_nadun_red.csv', 'raw_ravindu_red.csv']

    train_data = np.concatenate((   getArray(1, red_csv[0]),    getArray(1, red_csv[1]),    getArray(1, red_csv[2]),    
                                    getArray(2, green_csv[0]),  getArray(2, green_csv[1]),  getArray(2, green_csv[2]),  
                                    #getArray(3, blue_csv[0]),   getArray(3, blue_csv[1]),   getArray(3, blue_csv[2])   
                                ), axis=0)

    test_data = np.concatenate((    getArray(1, red_csv[3]),    getArray(1, red_csv[4]), 
                                    getArray(2, green_csv[3]),  getArray(2, green_csv[4]),
                                    #getArray(3, blue_csv[3]),   getArray(3, blue_csv[4])
                               ), axis=0)

    '''

    train_data = np.loadtxt('train_fourier2.csv' , delimiter=',',)
    test_data = np.loadtxt('test_fourier2.csv' , delimiter=',',)

    print "TRAIN DATA"
    print train_data.shape

    print "TEST DATA"
    print test_data.shape

    #np.savetxt("train.csv", train_data, delimiter=",")
    #np.savetxt("test.csv", test_data, delimiter=",")

    train_X = train_data[:,:-1]
    train_Y = train_data[:,-1]

    test_X = test_data[:,:-1]
    test_Y = test_data[:,-1]

    # FOR CROSS VALIDATION
    X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.4, random_state=0)
    
    #predicted = cross_val_predict(clf, train_X, train_Y, cv=10)
    #print("Cross-validation accuracy: ", metrics.accuracy_score(train_Y, predicted))


    #############################################################################################
    '''
    X_features = RBFSampler(gamma=1, random_state=1).fit_transform(train_X)
    X_testFeatures = RBFSampler(gamma=1, random_state=1).fit_transform(test_X)

    clf = SGDClassifier()
    clf.fit(X_features, train_Y)
    print(clf.score(X_testFeatures, test_Y)) 
    print(confusion_matrix(test_Y, clf.predict(X_testFeatures)))
    '''
    
    classifiers = [ 
                    SVC(decision_function_shape='ovo'), 
                    #SVC(kernel='linear', C=1),
                    DecisionTreeClassifier(),
                    KNeighborsClassifier(n_neighbors=9),
                    GaussianNB(),
                    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=3),
                    NearestCentroid(),
                    RandomForestClassifier()
                  ]

    for clf in classifiers:
        print ""
        print clf
        clf.fit(train_X, train_Y)
        #clf.fit(X_train, Y_train)                                                           
        print clf.classes_
        print(clf.score(test_X, test_Y)) 
        #print(clf.score(X_test, Y_test))                                              
        print(confusion_matrix(test_Y, clf.predict(test_X)))
        #print(confusion_matrix(Y_test, clf.predict(X_test))) 
 
    
    
    
if __name__ == "__main__":
    main()