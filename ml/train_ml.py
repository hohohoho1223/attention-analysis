import os
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from ml.classification_ml  import Classification_ML

MODE = "vid"

def scalingDF(df):
    # scaler = QuantileTransformer(output_distribution='normal')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    joblib.dump(scaler, f"./ml/trained_models_{MODE}/scaler_{MODE}.joblib") 
    df_s = pd.DataFrame(scaled,index=df.index, columns=df.columns)
    return df_s

def train():
    df = pd.read_csv(f"./features_{MODE}.csv")
    if MODE == "vid":
        test_df = pd.read_csv(f"./features_{MODE}_test.csv")

    df0 = df[df['Label'] == 0]
    df1 = df[df['Label'] == 1]
    df2 = df[df['Label'] == 2]

    #클래스 불균형 처리 : Resampling for treating the class imabalnce problem
    n = min(len(df0),len(df1),len(df2))
    df0_ds = resample(df0, replace=True, n_samples=n, random_state=42)
    df1_ds = resample(df1, replace=True, n_samples=n, random_state=42)
    df2_ds = resample(df2, replace=True, n_samples=n, random_state=42)

    df = pd.concat([df0_ds,df1_ds,df2_ds])
    df = df.sample(frac=1)

    if MODE == 'img':
        df_x = df.loc[:, "x0":"gaze_ratio"]#마지막 feature 컬럼까지
        df_y = df.loc[:,"Label"]
        df = pd.concat([scalingDF(df_x),df_y],axis=1)
    else:
        drop_cols = ['Label', 'vid_id', 'window_id']
        df_x = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df_y = df.loc[:,"Label"]
        df = pd.concat([scalingDF(df_x),df_y],axis=1)

        # test는 scaler만 사용
        testdf_x = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
        testdf_y = test_df.loc[:,"Label"]
        scaler = joblib.load(f"./ml/trained_models_{MODE}/scaler_{MODE}.joblib")
        testdf_x_scaled = pd.DataFrame(scaler.transform(testdf_x), index=testdf_x.index, columns=testdf_x.columns)
        test_df = pd.concat([testdf_x_scaled, testdf_y], axis=1)

    #create an object for Classification_ML
    obj = Classification_ML()

    '''Generate the training data for the four feature sets: 
        S1(facial landmark feature)
        S2(Eye gaze and head pose features)
        S3(AU features)
        S4(Combined features)'''

    if MODE == "img":
        training_data=obj.feature_sets(df, mode=MODE)
    else:
        training_data=obj.feature_sets(df, mode=MODE, test_df=test_df)

    dfML_result=[]
    roc_plot=[]
    cm_plot=[]

    path_trainedmodel = f"./ml/trained_models_{MODE}"

    #train the feature sets with our classifiction models
    for i in range(4):
        temp , plot, plot_cm= obj.classifier_result(training_data[i],i,path_trainedmodel)
        print(temp)
        #save the result in dfML_result dataframe
        dfML_result.append(pd.DataFrame(temp,columns=["model","Accuracy","Precision","Recall","F-measure"],index=["XGBoost", "Random forest"]).round(decimals=2))
        roc_plot.append(plot)
        cm_plot.append(plot_cm)

    #save the results and the plots
    os.makedirs(f'./ml/results_{MODE}', exist_ok=True)
    for i in range(4):
        dfML_result[i].to_csv(f'./ml/results_{MODE}/result_s{i}.csv')
        roc_plot[i].savefig(f'./ml/results_{MODE}/roc_plot_s{i}.pdf')
        cm_plot[i].savefig(f'./ml/results_{MODE}/cm_plot_s{i}.pdf')