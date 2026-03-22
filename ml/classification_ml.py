import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize


class Classification_ML():
  def stratified_split(self, df, val_percent=0.2):
    '''
    Function to split a dataframe into train and validation sets, while preserving the ratio of the labels in the target variable
    Inputs:
    - df, the dataframe
    - target, the target variable
    - val_percent, the percentage of validation samples, default 0.2
    Outputs:
    - train_idxs, the indices of the training dataset
    - val_idxs, the indices of the validation dataset
    '''
    classes=[0,1,2]
    train_idxs, val_idxs = [], []
    for c in classes:
        idx=list(df.loc[df['Label']==c].index)
        np.random.seed(42) # 재현성 보장
        np.random.shuffle(idx)
        val_size=int(len(idx)*val_percent)
        val_idxs+=idx[:val_size]
        train_idxs+=idx[val_size:]
    return train_idxs, val_idxs

  def feature_sets(self, df, mode, test_df=None): # mode, test_df 지정 추가
    if mode == 'img':
        return self._feature_sets_img(df)
    else:
        return self._feature_sets_vid(df, test_df=test_df)

  def classifier_result(self, training_data,i,path_o):
    temp=[]

    #initial model
    xgb = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,
                        max_depth=9,
                        min_child_weight=1,
                        objective='multi:softmax',
                        num_class=3,
                        seed=27)
    xgb.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
    predicted_labels = xgb.predict(training_data['X_test'])
    temp.append([xgb, accuracy_score(training_data['Y_test'], predicted_labels),
                 precision_score(training_data['Y_test'], predicted_labels, average='weighted'), 
                 recall_score(training_data['Y_test'], predicted_labels, average='weighted'),
                 f1_score(training_data['Y_test'], predicted_labels,average='micro')])     
    path_d = os.path.join(path_o,"model_xgb_%d.joblib"%(i))   
    joblib.dump(xgb,path_d)

    #rf_clf
    rf_clf = RandomForestClassifier(n_jobs=None,random_state=27, verbose=1)
    rf_clf.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
    predicted_labels = rf_clf.predict(training_data['X_test'])
    # train_pred = rf_clf.predict(training_data['X_train'])
    temp.append([rf_clf, accuracy_score(training_data['Y_test'], predicted_labels),
                 precision_score(training_data['Y_test'], predicted_labels, average='weighted'), 
                 recall_score(training_data['Y_test'], predicted_labels, average='weighted'),
                 f1_score(training_data['Y_test'], predicted_labels,average='micro')])  
    path_d = os.path.join(path_o,"model_rf_%d.joblib"%(i))   
    joblib.dump(rf_clf,path_d)
    
    
    fig = self.plot_roc_curve_all([xgb,rf_clf],training_data['X_test'],training_data['Y_test'])
    fig_cm  = self.plot_confusion_matrix([xgb, rf_clf], training_data['X_test'], training_data['Y_test'])

    return(temp, fig, fig_cm)

  def roc_values(self, y_score,y_test,n_classes):

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr,tpr,roc_auc
  

  def plot_roc_curve_all(self, model_list, X_test, Y_test):
    
    m = len(model_list)
    Y_score=[]
    for i in range(m):
      Y_score.append(model_list[i].predict_proba(X_test))

    n_classes = len(np.unique(Y_test))
    Y_test = label_binarize(Y_test, classes=np.arange(n_classes))
    Y_test = np.array(Y_test)
    
    fpr=[]
    tpr=[]
    roc_auc=[]
    for i in range(m):
      f, t, r = self.roc_values(Y_score[i],Y_test,n_classes)
      fpr.append(f)
      tpr.append(t)
      roc_auc.append(r)

    # Plot ROC curve
    fig = plt.figure(figsize=(5,5))
    plt.rc('axes', labelsize=18)
    plt.rc('legend', fontsize=12)
    plt.plot(fpr[0]["micro"], tpr[0]["micro"],color="red",
            label='XGBoost(area = {0:0.2f})'
                  ''.format(roc_auc[0]["micro"]))
    plt.plot(fpr[1]["micro"], tpr[1]["micro"],color="blue",
            label='Random forest(area = {0:0.2f})'
                  ''.format(roc_auc[1]["micro"]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    return fig
    

  def plot_roc_curve(self, model, X_test, Y_test):
   
    y_score = model.predict_proba(X_test)

    n_classes = len(np.unique(Y_test))
    y_test = label_binarize(Y_test, classes=np.arange(n_classes))
    y_test = np.array(y_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot ROC curve
    fig = plt.figure(figsize=(4,4))
    # plt.plot(fpr["micro"], tpr["micro"],
    #         label='micro-average ROC curve (area = {0:0.2f})'
    #               ''.format(roc_auc["micro"]))
    # plt.plot(fpr["macro"], tpr["macro"],
    #         label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='Class {0} (area = {1:0.2f})'
                                      ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    return fig
  
  def plot_confusion_matrix(self, model_list, X_test, Y_test):
      fig, axes = plt.subplots(1, 2, figsize=(12, 5))
      labels = ['Unfocus', 'Partial', 'Focus']
      names = ['XGBoost', 'Random Forest']
      
      for i, (model, ax) in enumerate(zip(model_list, axes)):
          predicted = model.predict(X_test)
          cm = confusion_matrix(Y_test, predicted)
          disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
          disp.plot(ax=ax)
          ax.set_title(names[i])
      
      plt.tight_layout()
      plt.show()
      return fig


  ################################
  # feature img/vid
  ################################

    
  def _feature_sets_img(self, df):
    # data split
    train_idxs, val_idxs = self.stratified_split(df, val_percent=0.25)
    val_idxs, test_idxs = self.stratified_split(df[df.index.isin(val_idxs)], val_percent=0.5)

    train_df = df[df.index.isin(train_idxs)]
    val_df = df[df.index.isin(val_idxs)]
    test_df = df[df.index.isin(test_idxs)]

    # 특성에 따른 ML 성능
    def get_sets(data):
        s1 = data.loc[:,'x0':'y467'] # 랜드마크
        s2 = data[['pitch', 'yaw', 'roll']] # head pose
        s3 = data[['gaze_ratio']]
        s4 = pd.concat([s1,s2,s3], axis=1) # 전체 통합
        return s1.values, s2.values, s3.values, s4.values
    
    X_s1_train, X_s2_train, X_s3_train, X_s4_train = get_sets(train_df)
    X_s1_val, X_s2_val, X_s3_val, X_s4_val = get_sets(val_df)
    X_s1_test, X_s2_test, X_s3_test, X_s4_test = get_sets(test_df)

    Y_train = train_df[['Label']].values
    Y_val = val_df[['Label']].values
    Y_test = test_df[['Label']].values

    #store data, all in numpy arrays
    training_data=[]
    for X_train, X_val, X_test in [
        (X_s1_train, X_s1_val, X_s1_test),
        (X_s2_train, X_s2_val, X_s2_test),
        (X_s3_train, X_s3_val, X_s3_test),
        (X_s4_train, X_s4_val, X_s4_test),
    ]:
        training_data.append({
            'X_train': X_train, 'Y_train': Y_train,
            'X_val':   X_val,   'Y_val':   Y_val,
            'X_test':  X_test,  'Y_test':  Y_test
        })
    return training_data

  def _feature_sets_vid(self, df, test_df):
    # data split (타 데이터 사용 시)
    # train_idxs, val_idxs = self.stratified_split(df, val_percent=0.25)
    # val_idxs, test_idxs = self.stratified_split(df[df.index.isin(val_idxs)], val_percent=0.5)

    # train_df = df[df.index.isin(train_idxs)]
    # val_df = df[df.index.isin(val_idxs)]
    # test_df = df[df.index.isin(test_idxs)]

    train_df = df
    test_df = test_df

    # 특성에 따른 ML 성능
    def get_sets(data):
        s1 = data.filter(regex=r'^(x|y)\d+_(mean|std)$')  # 랜드마크 통계
        s2 = data[['pitch_mean','pitch_std','yaw_mean','yaw_std','roll_mean','roll_std']] # head pose
        s3 = data[['gaze_mean','gaze_std','blink_mean']]
        s4 = pd.concat([s1,s2,s3], axis=1) # 전체 통합
        return s1.values, s2.values, s3.values, s4.values
    
    X_s1_train, X_s2_train, X_s3_train, X_s4_train = get_sets(train_df)
    # X_s1_val, X_s2_val, X_s3_val, X_s4_val = get_sets(val_df)
    X_s1_test, X_s2_test, X_s3_test, X_s4_test = get_sets(test_df)

    Y_train = train_df[['Label']].values
    # Y_val = val_df[['Label']].values
    Y_test = test_df[['Label']].values

    #store data, all in numpy arrays
    training_data=[]
    # for X_train, X_val, X_test in [ # val 있는 경우
    #     (X_s1_train, X_s1_val, X_s1_test),
    #     (X_s2_train, X_s2_val, X_s2_test),
    #     (X_s3_train, X_s3_val, X_s3_test),
    #     (X_s4_train, X_s4_val, X_s4_test),
    # ]:
    for X_train, X_test in [ # val 없음
        (X_s1_train, X_s1_test),
        (X_s2_train, X_s2_test),
        (X_s3_train, X_s3_test),
        (X_s4_train, X_s4_test),
    ]:
        training_data.append({
            'X_train': X_train, 'Y_train': Y_train,
            # 'X_val':   X_val,   'Y_val':   Y_val,
            'X_test':  X_test,  'Y_test':  Y_test
        })
    return training_data
    