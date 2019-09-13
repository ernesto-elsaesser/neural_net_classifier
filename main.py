# (c) Ernesto Elsäßer 2019

import classifier

wc = classifier.WeightClassifier()

train_filename = input("data set for training (CSV): ")
wc.load_data(train_filename)
wc.train()

test_filename = input("data set for testing (CSV): ")
wc.load_data(test_filename)
verbose = input("print classifications (y/n)? ") == "y"
wc.test(verbose = verbose)
