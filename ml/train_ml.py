import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from ml.ml_classification  import ML_classification

def scalingDF(df):
    # scaler = QuantileTransformer(output_distribution='normal')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df_s = pd.DataFrame(scaled,index=df.index, columns=df.columns)
    return df_s

def train():
    df = pd.read_csv('./features.csv')
    df0 = df[df['Label'] == 0]
    df1 = df[df['Label'] == 1]
    df2 = df[df['Label'] == 2]

    #클래스 불균형 처리 : Resampling for treating the class imabalnce problem
    n = min(len(df0),len(df1),len(df2))
    df1_ds = resample(df1, replace=True, n_samples=n, random_state=42)
    df2_ds = resample(df2, replace=True, n_samples=n, random_state=42)

    df = pd.concat([df0,df1_ds,df2_ds])

    df = df.sample(frac=1)
    df_x = df.loc[:,"x0":"blink"] #마지막 feature 컬럼까지
    df_y = df.loc[:,"Label"]

    df = pd.concat([scalingDF(df_x),df_y],axis=1)

    #create an object for ML_classification
    obj = ML_classification()

    '''Generate the training data for the four feature sets: 
        S1(facial landmark feature)
        S2(Eye gaze and head pose features)
        S3(AU features)
        S4(Combined features)'''

    training_data=obj.feature_sets(df)

    dfML_result=[]
    roc_plot=[]

    path_trainedmodel = "./ml/trained_models"

    #train the feature sets with our classifiction models
    for i in range(4):
        temp , plot= obj.classifier_result(training_data[i],i,path_trainedmodel)
        print(temp)
        #save the result in dfML_result dataframe
        dfML_result.append(pd.DataFrame(temp,columns=["model","Accuracy","Precision","Recall","F-measure"],index=["XGBoost", "Random forest"]).round(decimals=2))
        roc_plot.append(plot)

    #save the results and the plots
    os.makedirs('./ml/results', exist_ok=True)
    for i in range(4):
        dfML_result[i].to_csv(f'./ml/results/result_s{i}.csv')
        roc_plot[i].savefig(f'./ml/results/roc_plot_s{i}.pdf')