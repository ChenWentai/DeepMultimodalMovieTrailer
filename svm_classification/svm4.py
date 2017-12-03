import numpy as np
import os
from sklearn import svm
from sklearn import cross_validation,grid_search
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
def clean_data(file_path='/output/splits/demo/train'):
    c = os.listdir(file_path)
    file_list = []
    for f in c:
        if '.npy' in f:
            file_list.append(f)
    file_list = sorted(file_list)
    fea = []
    for f in file_list:
        fea.append(np.load(file_path+f))
    #average
    fea = np.array(fea)
    fea_ave = []
    for item in fea:
        ave_item = list(np.mean(item, axis = 0))
        fea_ave.append(ave_item)
#     #concatenate in 1-d array
#     fea_1d = []
#     for i in range(len(fea)):
#         print (np.shape(fea[i]))
#         length = len(fea[i])
#         width = len(fea[i][0])
#         fea_1d.append(np.reshape(fea[i],length*width))    
#     fea = fea_1d
     #specify the classes to each feature
    labels = []
    for f in file_list:
        if 'action' in f:
            labels.append(0)
        elif 'drama' in f:
            labels.append(1)
        elif 'horror' in f:
            labels.append(2)
        elif 'romance' in f:
            labels.append(3)
        elif 'sci-fi' in f:
            labels.append(4)
        else:
            print ('file %s does not conforms with the naming code '%f)
            break   
    return fea_ave,labels,file_list

    
def isEqual(objects):
    '''If all the elements in the list objects is equal, return True; otherwise return False'''
    for i in range(0,len(objects)-1):
        if objects[i] != objects[i+1]:
            return False
    return True    
def numerize(genre):
    '''Convert genre to number,0:action,1:drama,2:horror,3:romance,4:sci-fi'''
    lookup = ['action','drama','horror','romance','sci-fi']
    for i in lookup:
        if genre == i:
            return lookup.index(genre)
    return False 
def group_info(name,predict):
    #Integrate all the information
    '''
        return a list of infomation of each splits,
        list element format:[trailer index, splits index, true label, predicted label],
        every item is integer,
        size of info is equal to the number of splits.
        
    '''
    if not isEqual([len(name),len(predict)]):
        print('vector length not match!')
        return
    info = []
    for i in range(len(name)):
        text = name[i].split('-')
        #remove the whole trailer in splits
        if 'sci-fi' in name[i]:
            text = [text[0],'-'.join(text[1:3]),text[-1]]
            
        try:
            text[0] = int(text[0])
            t = numerize(text[1])
            text[1] = int(text[2][0:5])
            text[2] = t
        except:
            info.append('')
            continue
        info.append(text)
        info[i].append(predict[i])
        #print (info)
    info = list(filter(None, info))
    return info

 
def group(info,N):
    temp = [] 
    result = []
    t = 0
    try:
        for j in range(0,N):
            for i in range(t,len(info)-1):
                temp.append(info[i])
                if (info[i][0] != info[i+1][0]) or (info[i][2] != info[i+1][2]):
                    t = i+1
                    break
            result.append(temp)
            temp = []
    except:
        print ('N is not correct!')
        return False
    return result
def major_vote(trailer):
    vote = [0,0,0,0,0]
    for item in trailer:
        vote[item[3]] += 1
    return vote.index(max(vote))
def get_N(info):
    '''
        This function is used to find the numebr of trailers, used in the function 'group'
    '''
    N = 0
    for i in range(len(info)-1):
        if (info[i][0]!=info[i+1][0]) or (info[i][2]!=info[i+1][2]) :
            N = N +1
    return N+1

def train_svm_classifier(train_fea,train_labels): 
    param = [
    {
        "kernel": ["linear"],
        "C": [100],

    }
    ]
    #train the SVM classifier
    svm = SVC(probability = True)
    clf = grid_search.GridSearchCV(svm,param,cv=10, n_jobs=4, verbose=3)
    classifier = clf.fit(train_fea,train_labels)
    return classifier
def CV_train_svm_classifier(features,labels):
    '''
    features: array of input features
    lables: array of labels associated with the input deatures
    model_output_path: path for storing the trained svm model
    '''
    test_prop = 0.2
    #save 20% 20 data for performance evaluation
    x_train,x_test,y_train,y_test = cross_validation.train_test_split(features, labels,test_size = test_prop)
    param = [
    {
        "kernel": ["linear"],
        "C": [1,10,100,1000]
    }
#    {
 #       "kernel": ["rbf"],
  #      "C": [10, 100],
   #     "gamma": [ 1e-5]
    #}
    ]


    #request probability estimation
    svm = SVC(probability = True)

    #10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = grid_search.GridSearchCV(svm,param,cv=10, n_jobs=4, verbose=3)

    clf.fit(x_train,y_train)
#     if os.path.exists(model_output_path):
#         joblib.dump(clf.best_estimator_,model_output_path)
#     else:
#         print("blabla")

    print("\nbest parameters set:")
    print(clf.best_params_)
    print("\ntest size:")
    print(test_prop)

    y_predict = clf.predict(x_test)

    labels = sorted(list(set(labels)))
    print("\nconfusion matrix:")
#    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(y_test,y_predict,labels = labels))

    print("\nclassification report:")
    print(classification_report(y_test,y_predict))

#test by majority vote
def test_majority(classifier, test_fea, test_labels, file_list):
    
    predict = classifier.predict(test_fea)
    info = group_info(file_list,predict)
    print ("info_size:",np.shape(info))
    N = get_N(info)
    print("N",N)	
    grouped_info = group(info,N)
    print ("grouped_info:",grouped_info) 
    print ("grouped_info_size",np.shape(grouped_info))
    trailer_labels = [item[0][2] for item in grouped_info]
    result = []

    for trailer in grouped_info:
        result.append(major_vote(trailer))
    labels = sorted(list(set(test_labels)))
    print('\nconfusion matrix:')
    print(confusion_matrix(trailer_labels,result,labels = labels))
    print("\nclassification report:")
    print(classification_report(trailer_labels,result))
    return predict,result
    
        
        

# #train SVM model
train_fea,train_labels,file_list_train = clean_data('./output/splits/demo/train/')
#CV_train_svm_classifier(train_fea,train_labels)
classifier = train_svm_classifier(train_fea,train_labels)
test_fea,test_labels,file_list_test = clean_data('./output/splits/demo/test/')
predict,result = test_majority(classifier,test_fea,test_labels,file_list_test)
predict = classifier.predict(test_fea)

# info = group_info(file_list,predict)
# N = get_N(info)
# grouped_info = group(info,N)
# trailer_labels = [item[0][2] for item in grouped_info]

# result = []
# for trailer in grouped_info:
#     result.append(major_vote(trailer))
# #print(result)
# labels = sorted(list(set(train_labels)))
# print('\nconfusion matrix:')
# print(confusion_matrix(trailer_labels,result,labels = labels))
# print("\nclassification report:")
# print(classification_report(test_labels,predict))


