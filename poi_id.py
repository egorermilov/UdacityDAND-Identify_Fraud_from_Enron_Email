import sys
import pickle
import random
import matplotlib
from matplotlib import pyplot
from numpy import mean
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV

# if True statement is being used very often here.
# It's just my personal preference, i don't like big chunks of code commented
# And I often need to turn on/off big parts of code during while debugging

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

if True:
    # We will add more features later
    features_list = ['poi']

    # dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )





if True:
    ### Task 2: Remove outliers

    ### EXPLORATION

    num_data_points = len(data_dict)
    num_data_features = len(data_dict[data_dict.keys()[0]])

    num_poi = 0
    for dic in data_dict.values():
        if dic['poi'] == 1:
            num_poi += 1

    print "Total number of data points: ", num_data_points
    print "Number of features used: ", num_data_features
    print "POIs: ", num_poi


if True:
    # Features with NA
    features_with_na = {}

    for dic in data_dict.values():
        for feature, value in dic.iteritems():
            if value == "NaN":
                features_with_na.update({feature: (features_with_na.get(feature,0)+1)})

    for key, value in features_with_na.items():
        print key + " = " + str(value)

if False:
    # Visualization
    for dic in data_dict.values():
        matplotlib.pyplot.scatter( dic['salary'] , dic['bonus']  )

    matplotlib.pyplot.xlabel("Salary")
    matplotlib.pyplot.ylabel("Bonus")
    matplotlib.pyplot.show()


if True:
    # there is an outlier
    for key, value in data_dict.items():
        if value['salary'] != 'NaN' and value['salary'] > 10000000:
            print key
    # we should remove TOTAL row from the salaries & bonuses list
    del data_dict["TOTAL"]
    del data_dict["THE TRAVEL AGENCY IN THE PARK"]
    print data_dict["LOCKHART EUGENE E"]

if True:
    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ## Communication:
    # fraction_to_poi = from_this_person_to_poi / from_messages
    # fraction_from_poi = from_poi_to_this_person / to_messages
    for item in my_dataset:
        person = my_dataset[item]
        if (all(
                [
                        person['from_poi_to_this_person'] != 'NaN',
                        person['from_this_person_to_poi'] != 'NaN',
                        person['to_messages'] != 'NaN',
                        person['from_messages'] != 'NaN'
                ]
        )):
            fraction_from_poi = float(person["from_poi_to_this_person"]) / float(person["to_messages"])
            person["fraction_from_poi"] = fraction_from_poi
            fraction_to_poi = float(person["from_this_person_to_poi"]) / float(person["from_messages"])
            person["fraction_to_poi"] = fraction_to_poi
        else:
            person["fraction_from_poi"] = person["fraction_to_poi"] = 0


if True:
    ## Financial:
    # wealth = salary + total_stock_value + exercised_stock_options + bonus
    for item in my_dataset:
        person = my_dataset[item]
        if (all([    person['salary'] != 'NaN',
                    person['total_stock_value'] != 'NaN',
                    person['exercised_stock_options'] != 'NaN',
                    person['bonus'] != 'NaN'
                ])):
            person['wealth'] = sum([person[field] for field in ['salary',
                                                               'total_stock_value',
                                                               'exercised_stock_options',
                                                               'bonus']])
        else:
            person['wealth'] = 'NaN'




if True:
    my_features = features_list + ['fraction_from_poi',
                                   'fraction_to_poi',
                                   'shared_receipt_with_poi',
                                   'expenses',
                                   'loan_advances',
                                   'long_term_incentive',
                                   'other',
                                   'restricted_stock',
                                   'restricted_stock_deferred',
                                   'deferral_payments',
                                   'deferred_income',
                                   'salary',
                                   'total_stock_value',
                                   'exercised_stock_options',
                                   'total_payments',
                                   'bonus',
                                   'wealth']

    ### Extract features and labels from dataset
    data = featureFormat(my_dataset, my_features, sort_keys = True)
    labels, features = targetFeatureSplit(data)


if True:
    # Feature scaling
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # K best features
    k_best = SelectKBest(k=5)
    k_best.fit(features, labels)

    results_list = zip(k_best.get_support(), my_features[1:], k_best.scores_)
    results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
    print "K best features:", results_list

    ## Features chosen by SelectKBest
    my_features = features_list + [
        'exercised_stock_options',
        'total_stock_value',
        'bonus',
        'salary',
        'fraction_to_poi',
        #'wealth',
        #'deferred_income',
        #'long_term_incentive',
        #'restricted_stock',
        #'total_payments'
    ]


    data = featureFormat(my_dataset, my_features, sort_keys = True)
    labels, features = targetFeatureSplit(data)




if True:
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    def test_clf(grid_search, features, labels, parameters, iterations=100):
        precision, recall, accuracy = [], [], []
        for iteration in range(iterations):
            features_train, features_test, labels_train, labels_test = \
                train_test_split(features, labels, random_state=iteration)
            grid_search.fit(features_train, labels_train)
            predictions = grid_search.predict(features_test)
            precision = precision + [precision_score(labels_test, predictions)]
            recall = recall + [recall_score(labels_test, predictions)]
            accuracy = accuracy + [accuracy_score(labels_test, predictions)]
            if iteration % 10 == 0:
                sys.stdout.write('.')
        m_precision = mean(precision)
        m_recall = mean(recall)
        #print '\nAccuracy:', m_accuracy
        print '\nPrecision:', m_precision
        print 'Recall:', m_recall
        best_params = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print '%s=%r, ' % (param_name, best_params[param_name])
        return m_precision, m_recall


# Test the effect of the aggregated features on the final algorithm performance
if False:
    aggr_test_features = features_list + [
        'shared_receipt_with_poi',
        'expenses',
        'loan_advances',
        'long_term_incentive',
        'other',
        'restricted_stock',
        'restricted_stock_deferred',
        'deferral_payments',
        'deferred_income',
        'salary',
        'total_stock_value',
        'exercised_stock_options',
        'total_payments',
        'bonus'
    ]  # 'wealth', 'fraction_from_poi', 'fraction_to_poi',

    from sklearn.naive_bayes import GaussianNB

    # ATTENTION (results for different aggregate features are the same)
    data = featureFormat(my_dataset, (aggr_test_features), sort_keys=True)
    labels, features = targetFeatureSplit(data)
    clf = GaussianNB()
    parameters = {}
    grid_search = GridSearchCV(clf, parameters)
    print '\nGaussianNB TEST (no aggr) ::'
    test_clf(grid_search, features, labels, parameters)

    data = featureFormat(my_dataset, (aggr_test_features + ['wealth']), sort_keys=True)
    labels, features = targetFeatureSplit(data)
    clf = GaussianNB()
    parameters = {}
    grid_search = GridSearchCV(clf, parameters)
    print '\nGaussianNB TEST (+wealth) ::'
    test_clf(grid_search, features, labels, parameters)

    data = featureFormat(my_dataset, (aggr_test_features + ['fraction_from_poi']), sort_keys=True)
    labels, features = targetFeatureSplit(data)
    clf = GaussianNB()
    parameters = {}
    grid_search = GridSearchCV(clf, parameters)
    print '\nGaussianNB TEST (+fraction_from_poi) ::'
    test_clf(grid_search, features, labels, parameters)

    data = featureFormat(my_dataset, (aggr_test_features + ['fraction_to_poi']), sort_keys=True)
    labels, features = targetFeatureSplit(data)
    clf = GaussianNB()
    parameters = {}
    grid_search = GridSearchCV(clf, parameters)
    print '\nGaussianNB TEST (+fraction_to_poi) ::'
    test_clf(grid_search, features, labels, parameters)



if False:
    my_features = features_list + [
        'exercised_stock_options',
        'total_stock_value',
        'bonus',
        'salary',
        'fraction_to_poi',
        'wealth',
        'deferred_income',
        'long_term_incentive',
        'restricted_stock',
        'total_payments'
    ]

    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier

    for k_features in range(2, 12):
        print "\nTESTING " + str(k_features-1) + " FEATURES "
        data = featureFormat(my_dataset, my_features[0:k_features], sort_keys=True)
        labels, features = targetFeatureSplit(data)
        clf = GaussianNB()
        parameters = {}
        grid_search = GridSearchCV(clf, parameters)
        print '\nGaussianNB ::'
        test_clf(grid_search, features, labels, parameters)





if False:
    # We decided to use only 9 features
    my_features = features_list + [
        'exercised_stock_options',
        'total_stock_value',
        'bonus',
        'salary',
        'fraction_to_poi',
        'wealth',
        'deferred_income',
        'long_term_incentive',
        'restricted_stock'
        # 'total_payments'
    ]

    data = featureFormat(my_dataset, my_features, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    parameters = {}
    grid_search = GridSearchCV(clf, parameters)
    print '\nGaussianNB ::'
    test_clf(grid_search, features, labels, parameters)




if False:
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    parameters = {}
    grid_search = GridSearchCV(clf, parameters)
    print '\nDecisionTree (NO tuning) ::'
    test_clf(grid_search, features, labels, parameters)

    clf = tree.DecisionTreeClassifier()
    parameters = {
        'min_samples_split': [2, 10, 20],
        'max_depth': [None, 2, 5, 10],
        'min_samples_leaf': [1, 5, 10],
        'max_leaf_nodes': [None, 2, 5, 10]
    }
    grid_search = GridSearchCV(clf, parameters)
    print '\nDecisionTree (tuning) ::'
    test_clf(grid_search, features, labels, parameters)



if False:
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier()
    parameters = {}
    grid_search = GridSearchCV(clf, parameters)
    print "\nKNeighborsClassifier  (NO tuning) ::"
    test_clf(grid_search, features, labels, parameters)

    # ATTENTION (precision and recall are lower here!)
    clf = KNeighborsClassifier()
    parameters = {'n_neighbors': [3, 5, 10],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['ball_tree', 'kd_tree', 'brute']}
    grid_search = GridSearchCV(clf, parameters)
    print "\nKNeighborsClassifier (tuning) ::"
    test_clf(grid_search, features, labels, parameters)




if True:
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info:
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    # Example starting point. Try investigating other evaluation techniques!
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    dump_classifier_and_data(clf, my_dataset, my_features)