import numpy as np
from flask import Flask, request,jsonify,render_template
import pickle
import sklearn 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
app = Flask(__name__)
model =pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    df_yield=pd.read_csv('yield_df.csv')
    df_yield.shape
    df_yield=df_yield.drop(['Unnamed: 0'],axis=1)
    df_yield.head()
    df_yield.groupby('Item').count()
    df_yield.groupby(['Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)
    df_yield.groupby(['Item','Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)
    import sklearn
    import seaborn as sns
    import matplotlib.pyplot as plt
    features=df_yield.loc[:,df_yield.columns!='hg/ha_yield']
    features =features.drop(['Year'],axis=1)
    label=df_yield['hg/ha_yield']
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct1=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
    # print("Features:", features)
    features=np.array(ct1.fit_transform(features))
    features
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    features[:,10]=le.fit_transform(features[:,10])
    yield_df_onehot=pd.DataFrame(features)
    yield_df_onehot["hg/ha_yield"]=label
    yield_df_onehot.head()
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    features=scaler.fit_transform(features)

    inputs =[x for x in request.form.values()]
    print(inputs)
    options =["Cassava","Maize","Potatoes","Rice,paddy","Sorghum","Soybeans","Wheat","Sweetpotatoes","Yams"]
    inputs[1]= options[int(inputs[1])]    
    inputs =np.array([inputs])
    inputs=np.array(ct1.transform(inputs))
    inputs[:,10]=le.transform(inputs[:,10])
    inputs=scaler.transform(inputs)
    prediction =model.predict(inputs)

    output =round(prediction[0],2)

    return render_template('index.html',prediction_text='Predicted Crop Yield: {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''    
    df_yield=pd.read_csv('yield_df.csv')
    df_yield.shape
    df_yield=df_yield.drop(['Unnamed: 0'],axis=1)
    df_yield.head()
    df_yield.groupby('Item').count()
    df_yield.groupby(['Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)
    df_yield.groupby(['Item','Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)
    import sklearn
    import seaborn as sns
    import matplotlib.pyplot as plt
    features=df_yield.loc[:,df_yield.columns!='hg/ha_yield']
    features =features.drop(['Year'],axis=1)
    label=df_yield['hg/ha_yield']
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct1=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
    # print("Features:", features)
    features=np.array(ct1.fit_transform(features))
    features
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    features[:,10]=le.fit_transform(features[:,10])
    yield_df_onehot=pd.DataFrame(features)
    yield_df_onehot["hg/ha_yield"]=label
    yield_df_onehot.head()
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    features=scaler.fit_transform(features)

    data =request.get_json(force=True)
    print(list(data.values()))
    inputs =(list(data.values()))
    print(inputs)
    options =["Cassava","Maize","Potatoes","Rice,paddy","Sorghum","Soybeans","Wheat","Sweetpotatoes","Yams"]
    inputs[1]= options[int(inputs[1])]    
    inputs =np.array([inputs])
    inputs=np.array(ct1.transform(inputs))
    inputs[:,10]=le.transform(inputs[:,10])
    inputs=scaler.transform(inputs)
    prediction =model.predict(inputs)
    output =prediction[0]
    returnjsonify(output)

if__name__=="__main__":
    app.run(debug=True)
