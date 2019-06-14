from mnist import MNIST
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

'''
We're using this code to compare dimensionality reduction. This uses a modified
MNIST dataset that normalizes all input images.
'''

def recognizePCA(train, trainlab, test, labels, num=None):

    if num is None:
        num=len(test)

    train4pca = np.array(train)
    test4pca = np.array(test)

    n_components = 20

    #fitting pca
    pca = PCA(n_components=n_components).fit(train4pca)

    xtrain = pca.transform(train4pca)

    xtest = pca.transform(test4pca)

    clf = KNeighborsClassifier()

    #fitting knn
    clf = clf.fit(xtrain, trainlab)

    #predicting
    y_pred = clf.predict(xtest[:num])



#checking the result and printing output
    r=0
    w=0
    for i in range(num):
        if y_pred[i] == labels[i]:
            r+=1
        else:
            w+=1
    print ("tested ", num, " digits")
    print ("correct: ", r, "wrong: ", w, "error rate: ", float(w)*100/(r+w), "%")
    print ("got correctly ", float(r)*100/(r+w), "%")

    return pca.components_

def normalize(ims):
	for image in range(len(ims)):
		for x in range(len(ims[image])):
			if ims[image][x] > 127:
				ims[image][x] = 255
			else:
				ims[image][x] = 0

mndata = MNIST()
trainims, trainlabels = mndata.load_training()
ims, labels = mndata.load_testing()

normalize(ims)
normalize(trainims)

recognizePCA(trainims, trainlabels, ims, labels)