
import streamlit as st
import pandas as pd
import numpy as np
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import base64
import random

import os


#import numpy.ndarray

#st.set_page_config(page_title='Prediction_app')


def st_pandas_to_csv_download_link(_df, file_name:str = "dataframe.csv"): 
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Download Dataset (CSV) </a>'
    st.markdown(href, unsafe_allow_html=True)
    
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    
def app(session_in):
    
    output_data = 'output' + session_in + '_1'
    output_model = 'output' + session_in + '_2'
    output_y = 'output' + session_in + '_3'
    
    aa = pd.DataFrame()

    with st.expander("Stage2. ?????? ?????? ???????????? - ?????????"):
        st.markdown("**Option1) ?????? ?????????**")
        st.markdown("Stage1. ????????? ?????? ?????? ?????? ???????????? Stage2.??? ?????? ???????????????.")
        st.markdown("**Option2) ?????? ?????????**")
        st.markdown("1. Stage1.?????? ????????? ??? ??????(????????????.pkl, Train?????????.xlsx)??? ???????????????.")
        st.markdown("2. Train ????????? ??????(.xlsx)??? ????????? ?????????. CTP, CTQ??? ?????? ?????? ???????????? ?????? ???????????????.")
        st.markdown("3. ???????????? ??????(.pkl)??? ????????? ?????????.")
        st.markdown("")
        st.markdown("**??????) CTP, CTQ ????????????**")
        st.markdown("1. ????????????(CTP)??? ???????????? ????????????(CTQ)??? ???????????????")
        st.markdown("2. ?????? ??????(CTQ)??? ????????????, ????????????(CTP) ????????? ????????? ?????? ??????????????? ???????????????.")
        st.markdown("3. ?????? ???????????? ????????????(CTP)??? ????????? ??????(.csv)??? ????????? ????????? ?????????. ????????? ????????? ?????? ????????? ????????? ??????(CTQ)??? ???????????????.")


    st.markdown("<h2 style='text-align: left; color: black;'>Stage2. ?????? ?????? ??????</h2>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.sidebar.write("")

    st.sidebar.subheader("????????? & ????????? ????????? ?????????")
    howtofetch = st.sidebar.radio("Select ???", ("Option1) ?????? ?????? ?????? ?????????", "Option2) ?????? ?????? ?????? ?????????"))

    if howtofetch == "Option1) ?????? ?????? ?????? ?????????":
        
        if output_data not in st.session_state:
            st.session_state[output_data] = aa
        df = st.session_state[output_data]
            
        if output_model not in st.session_state:
            st.session_state[output_model] = aa
        model = st.session_state[output_model]

        if output_y not in st.session_state:
            st.session_state[output_y] = aa
        outputY = st.session_state[output_y]
        outputY2 = -1 * outputY
        
        if len(df)==0:
            st.error("????????? ????????? ????????????!!")
            st.markdown("**_??? Stage1. ??? ?????? ???????????????._**")
            st.markdown("**_??? Stage1. [5. ?????? ?????????] ?????? ??? ????????? ??????????????????, ?????? ?????? ??? ????????? ????????? ?????????._**")
            st.markdown("**_??? [F5]??? ????????? ??? ???????????? ????????? ???????????? ?????? ??????????????????._**")
            
        else:
            
            #st.write(df)#?????????
            #st.write(model)#?????????
    
            st.subheader('**1. ???????????? ?????? ?????????**')
            st.write('')
            x = list(df.columns)
            
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.header('1. ?????? ?????????(.csv) ?????????')
            st.sidebar.markdown("_**- ?????? ????????? ??????**_")
            
            st.sidebar.write("")
            Selected_X = st.sidebar.multiselect('????????????(CTP) ????????????', x, x[:outputY2])
            
            y = [a for a in x if a not in Selected_X]
            
            Selected_y = st.sidebar.multiselect('????????????(CTQ) ????????????', y, y)
        
            Selected_X = np.array(Selected_X)
            Selected_y = np.array(Selected_y)
            
             
            st.write('**1.1 CTP ?????? :**',Selected_X.shape[0])
            st.info(list(Selected_X))
            st.write('')
        
            st.write('**1.2 CTQ ??????:**',Selected_y.shape[0])
            st.info(list(Selected_y))
        
            df2 = pd.concat([df[Selected_X],df[Selected_y]], axis=1)
            #df2 = df[df.columns == Selected_X]
        
            #Selected_xy = np.array((Selected_X,Selected_y))
            
            st.write('')
            st.write('')
            st.write('')
            st.write('')    
            st.write('')    
            
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.write("")
            
            st.sidebar.header('2. ????????????(.pkl) ?????????')
            st.sidebar.markdown("**_- ?????? ????????? ??????_**")
            
            st.subheader('**2. ???????????? ???????????? ??????**')
            st.write('')        
    
            st.markdown('**2.1 ????????? ???????????? ?????? :**')
            st.write(model)
            st.write('')
            st.write('')
            st.write('')
            
            st.markdown('**2.2 ?????? ?????? ????????? :**')
            X_train = df[Selected_X]
            y_train = df[Selected_y]
            
            scaler_y = StandardScaler().fit(y_train)
            predictions = model.predict(X_train)
        
            results = []
    
            msg = []
            mean = []
            std = []        
                
            kfold = KFold(n_splits=5, random_state=7, shuffle=True)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
            
            for i, element in enumerate(cv_results):
                if element <= 0.0:
                    cv_results[i] = 0.0
                    
                    
            results.append(cv_results)
            #    names.append(name)
            msg.append('%s' % model)
            mean.append('%f' %  (cv_results.mean()))
            std.append('%f' % (cv_results.std()))
                    
                    
            F_result3 = pd.DataFrame(np.transpose(msg))
            F_result3.columns = ['Machine_Learning_Model']
            F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
            F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
            
            #st.write(F_result3)    
    
            st.write('?????? ????????? ($R^2$):')
            
            R2_mean = list(F_result3['R2_Mean'].values)
            st.info( R2_mean[0] )
                
            st.write('?????? ????????? ?????? (Standard Deviation):')
            
            R2_std = list(F_result3['R2_Std'].values)
            st.info( R2_std[0])
                
            
            
            
                        
            
            df2 = df[Selected_X]
            columns = df2.columns
    
            test = []
            name = []
            #st.sidebar.write('3.1 Predict Single Condition')
            
    
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.write("")
            
            if Selected_y.shape[0] <= 1:
                st.sidebar.header('3. CTP, CTQ ??????')
                st.sidebar.write('**3.1 ????????????(CTQ) ??????**')
                
                #st.subheader('**3. Model Prediction **')
                #st.write('**3.1 Single Condition Prediction :**')
    
                for column in columns:
                        value = st.sidebar.number_input(column, None, None, df2[column].mean(),format="%.2f") #int(df2[column].mean()))
                
                        name.append(column)
                        test.append(value)
                
     
    
                
                st.write('')
                st.write('')
                st.write('')        
                st.write('')        
                st.write('')        
                              
                st.subheader('**3. CTP, CTQ ??????**')
     
                st.write('')        
                st.write('**3.1 ????????????(CTQ) ?????? - ?????? ?????????**')
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.1 ?????? ????????? ????????????(CTP)??? ?????? ?????? ??????(CTQ)??? ???????????????. </h6>", unsafe_allow_html=True)
                st.write('')        
    
                
                F_result = pd.DataFrame(name)
                F_result.columns = ['X Variables']
                F_result['Value'] = pd.DataFrame(test)
    
                
                #
     
            #para = st.sidebar.slider(para,mi,ma,cu,inter)
                #st.write('')
            
            
        
                #st.write(F_result)
                #scaler = StandardScaler().fit(X_train)
            
                if st.sidebar.button('CTQ ????????????'):
            
            #st.write(F_result)
                    F_result = F_result.set_index('X Variables')
                    F_result_T = F_result.T
    
                    #rescaled2 = scaler.transform(F_result_T)
        
                    predictions = model.predict(F_result_T)
        
                    st.markdown("_- ????????? ????????????(CTP) ??????_")
                    st.write(F_result)                    
    
                    st.write('**??? CTQ [', Selected_y[0],"]??? ?????? ?????????  :**" , predictions[0])
                    
                
     
                
                
                
                st.sidebar.write('')
                st.sidebar.write('')    
                st.sidebar.write('**3.2 ?????? ????????????(CTP) ??????**')
                
                ctq1 = df[Selected_y].mean()
                ctq2 = ctq1[0]
    
                
            #Target = st.sidebar.number_input(list(Selected_y)[0], 0.00, 1000.00, 0.00,format="%.2f")
                st.sidebar.write('**Step1) CTQ ?????? ??????????**')
                N_sample = 0
                Target = st.sidebar.number_input(Selected_y[0], None, None, ctq2,format="%.2f")
            
                st.sidebar.write("")    
                st.sidebar.write('**Step2) ????????? ?????? ??????????**')
                N_sample = st.sidebar.number_input("?????? ??????",0, 1000, 50, format="%d")
                
                name2=[]
                test2=[]
                count = 0
                
                st.sidebar.write("")    
                st.sidebar.write('**Step3) CTP ?????? ????????????**')
                
                for column in columns:
                
                    max1 = round(float(df[column].max()),3)
                    min1 = round(float(df[column].min()),3)
                
                    #rag1 = round(min1+((max1-min1)*0.1),3)
                    #rag2 = round(min1+((max1-min1)*0.9),3)
                
                    step = round((max1-min1)/20.0,3)
                
                    value = st.sidebar.slider(column, min1, max1, (min1,max1), step)
                         
                    name2.append(column)
                    test2.append(value)
                #param2.append(para_range)
                #st.write(min1,rag1,rag2,max1)
                #st.write(column)
                #st.write(test2)
                st.write('')        
                st.write('')        
                st.write('')
                st.write('**3.2 ?????? ????????????(CTP) ??????**')
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.2 ?????? ????????? CTQ ????????? ?????? ?????? ????????????(CTP)??? ???????????????. </h6>", unsafe_allow_html=True)
        
                if st.sidebar.button('CTP ????????????',key = count): 
                
                    para = []
                    para2 = []
                    para4 = []
                
                    #st.write(test2)
                    import itertools
                
                    for para in test2:
                        if para[0] == para[1]:
                            para = itertools.repeat(para[0],100)
                        else:
                            para = np.arange(round(para[0],6), round(para[1]+((para[1]-para[0])/100.0),6), round((para[1]-para[0])/100.0,6))
                    #st.write(para)
                        para2.append(para)
                
               
                    Iter = N_sample
                
                    para2 = pd.DataFrame(para2)
                    para2 = para2.T
                    para2 = para2.dropna().reset_index()
                    para2.drop(['index'],axis=1,inplace=True)
                
                    Iter2 = para2.shape[1]
                
                
    
                #st.write(Iter,Iter2)
                    for i in range(Iter):
                        para3 = []
                        para5 = []
                        for j in range(Iter2):
                            #st.write(i,j,list(para2[j]))
                            para3.append(random.sample(list(para2[j]),1))
                            para5.append(para3[j][0])
                        
                    #para3 = pd.DataFrame(para3).values
                    #para4.append(para3)
                        para4.append(para5)
                    
                    
                #para4 = pd.DataFrame(para4)
                    para4 = pd.DataFrame(para4)
                
                
                    para4.columns=list(Selected_X)
                
                    #st.write('_????????? ?????? ??????????????? ?????? :_')
                    #st.write(para4)
                
                    datafile = para4
    
                    #rescaled = scaler.transform(datafile)
            
                    predictions2 = model.predict(datafile)
    
                    para4['predicted results'] = predictions2
                
                #st.write(para4)
                
                    para4.sort_values(by='predicted results', ascending=True, inplace =True)
                
                    para4 = para4.reset_index()
                    para4.drop(['index'],axis=1,inplace=True)
                
                #st.write(para4)
                
                    def find_nearest(array, value):
                        array = np.asarray(array)
                        idx = (np.abs(array - value)).argmin()
                        return array[idx]
                
                    opt_result = find_nearest(para4['predicted results'],Target)
                
                
                #st.write(opt_result)
    
                    st.write('')
                    st.markdown("_- ????????? ?????? ?????? : %.2f_" %Target)
                    st.markdown("_- ????????? ?????? ?????? : %d_" %N_sample)
                    st.write('')               
                
                    df_max = para4[para4['predicted results']==para4['predicted results'].max()]
                    df_min = para4[para4['predicted results']==para4['predicted results'].min()]
                    df_opt = para4[para4['predicted results']==opt_result]
                    st.write('**??? ?????? ????????????(CTP) :**')
                    st.write(df_opt)
                    st.write("")
                    st.write('**??? ?????? CTQ ???????????? :**')
                    st.write(df_max)
                    st.write("")
                    st.write('**??? ?????? CTQ ???????????? :**')
                    st.write(df_min)
                    #st.info(list(Selected_X2))
                    st.write("")
                    st.write('**??? ?????? ?????? ?????? ?????? :**')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.write(para4)
                    st.write('')
         
                    #sample_list_1 = para4
                    #del sample_list_1["predicted results"]
                    #st.write(sample_list_1)
    
    
                    opt_ctq_best1 = df_opt['predicted results']
    
           
                    if len(opt_ctq_best1.index) == 1:
                        
                        plt.scatter(para4.index, para4.iloc[:,-1],color='#0e4194')
                        plt.scatter(opt_ctq_best1.index, opt_ctq_best1.iloc[0], color='#e30613')
                        sns.lineplot(x=para4.index, y=Target, color='red')
                        
                        st.pyplot()
                        
                        
                    else:
                        list1 = opt_ctq_best1.index
                        list1 = np.mean(list1)
                    
                        plt.scatter(para4.index, para4.iloc[:,-1],color='#0e4194')
                        plt.scatter(list1, opt_ctq_best1.iloc[0], color='#e30613')
                        sns.lineplot(x=para4.index, y=Target, color='red')
                        
                        st.pyplot()
                        
                
                
                    count +=1
    
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown('**??? ?????? ?????? ????????????**')
        
                    st_pandas_to_csv_download_link(para4, file_name = "Predicted_CTPs_Results.csv")
                    st.caption("**_??? ?????? ?????? ?????? ?????? : ????????? ????????? ??? [?????? ???????????? ?????? ??????]_**")
                    st.write('')
                    st.write('')
    
    
            
        
        
                st.sidebar.write('')
                st.sidebar.write('')    
            
        
                st.sidebar.write("**3.3 ????????????(CTQ) ?????? (Multi-case)**")
                uploaded_file3 = st.sidebar.file_uploader("Multi-case ?????????(.csv) ?????????", type=["csv"])
                if uploaded_file3 is not None:
                    def load_csv():
                        csv = pd.read_csv(uploaded_file3)
                        return csv
                
    
    
    
        
                count +=1
                st.write('')        
                st.write('')        
                st.write('')        
                st.write('**3.3 ????????????(CTQ) ?????? (Multi-case)**')
                
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.3 ?????? ???????????? ????????? ???????????? ?????????(CTP)??? ?????? ?????? ??????(CTQ)??? ???????????????. </h6>", unsafe_allow_html=True)
            
                if st.sidebar.button('CTQ ????????????',key = count):
                    df3 = load_csv()            
                    datafile = df3
    
                    #rescaled = scaler.transform(datafile)
                    
                    predictions2 = model.predict(datafile)
    
                    df3['predicted results'] = predictions2
    
                    
                
                    df_max = df3[df3['predicted results']==df3['predicted results'].max()]
                    df_min = df3[df3['predicted results']==df3['predicted results'].min()]
                    st.write("")
                    st.write('**??? ?????? CTQ ???????????? :**')
                    st.write(df_max)
                    st.write('**??? ?????? CTQ ???????????? :**')
                    st.write(df_min)
                    #st.info(list(Selected_X2))
                
                    st.write('')
                    st.write('**??? ?????? ?????? ?????? ?????? :**')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.write(df3)
                    sns.scatterplot(x=df3.index,y=df3.iloc[:,-1],s=30,color='#e30613')
                    st.write('')
                    st.pyplot()
                    count +=1
        
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown('**??? ?????? ?????? ???????????? (Multi-case)**')
        
                    st_pandas_to_csv_download_link(df3, file_name = "Predicted_CTQs_Results.csv")
                    st.caption("**_??? ?????? ?????? ?????? ?????? : ????????? ????????? ??? [?????? ???????????? ?????? ??????]_**")
                    
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
    
    
                
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
    
            #st.write(mi)
        #parameter_n_neighbors = st.sidebar.slider('Number of neighbers', 2, 10, (1,6), 2)  
    
        
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    
    
        
            else :
                
                df_y = df[Selected_y]
                
                st.sidebar.header('3. CTP, CTQ ??????')
                st.sidebar.write('**3.1 ????????????(CTQ) ??????**')
                
                #st.subheader('**3. Model Prediction **')
                #st.write('**3.1 Single Condition Prediction :**')
    
                for column in columns:
                        value = st.sidebar.number_input(column, None, None, df2[column].mean(),format="%.2f") #int(df2[column].mean()))
                
                        name.append(column)
                        test.append(value)
                
     
    
                
                st.write('')
                st.write('')
                st.write('')        
                st.write('')        
                st.write('')        
                              
                st.subheader('**3. CTP, CTQ ??????**')
                st.write('')        
                st.write('**3.1 ????????????(CTQ) ?????? - ?????? ?????????**')
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.1 ?????? ????????? ????????????(CTP)??? ?????? ?????? ??????(CTQ)??? ???????????????. </h6>", unsafe_allow_html=True)
                F_result = pd.DataFrame(name)
                F_result.columns = ['X Variables']
                F_result['Value'] = pd.DataFrame(test)
                
                
                
                #st.write(F_result)
     
            #para = st.sidebar.slider(para,mi,ma,cu,inter)
                st.write('')
            
            
        
                
                
            
                if st.sidebar.button('CTQ ????????????'):
            
            #st.write(F_result)
                    F_result = F_result.set_index('X Variables')
                    F_result_T = F_result.T
    
      
                    predictions = model.predict(F_result_T)
        
    
                    #predictions = pd.DataFrame(predictions[0],columns=['Value'])
                    #st.write(predictions)
                    
                    predictions2 = pd.DataFrame()
                    predictions2['Y Variable'] = df[Selected_y].columns
                    predictions2['Value'] = pd.DataFrame(predictions[0])
                    
                    
                    #Selected_y_list = Selected_y.tolist()
                    #st.write(Selected_y_list)
                    
                    st.write("_????????? ????????????(CTP) ??????_")
                    st.write(F_result)
                    
                    st.write("**??? CTQ(Y variables)??? ?????? ????????? :**")
                    st.write(predictions2)
                    
                    
                
                
                
                st.sidebar.write('')
                st.sidebar.write('')    
                st.sidebar.write('**3.2 ?????? ????????????(CTP) ??????**')
            
            #Target = st.sidebar.number_input(list(Selected_y)[0], 0.00, 1000.00, 0.00,format="%.2f")
                st.sidebar.write('**Step1) CTQ ?????? ??????????**')
                
                sample_target = df[Selected_y].shape[1]
                                
                Target = []
                
    
                for i in range(sample_target):
    
                    Sel_y_mean = df[Selected_y[i]]
                    Sel_y_mean2 = Sel_y_mean.mean()
     
                    Target.append(st.sidebar.number_input(Selected_y[i], None, None, Sel_y_mean2, format="%.2f"))
            
            
            
                N_sample = 0
                st.sidebar.write('')
                st.sidebar.write('**Step2) ????????? ?????? ??????????**')
                N_sample = st.sidebar.number_input('?????? ??????',0,1000,50, format="%d")
                
            
                
                name2=[]
                test2=[]
                count = 0
                
                st.sidebar.write('')
                st.sidebar.write('**Step3) CTP ?????? ????????????**')
                for column in columns:
                
                    max1 = round(float(df[column].max()),3)
                    min1 = round(float(df[column].min()),3)
                    
                    
                    #rag1 = round(min1+((max1-min1)*0.1),3)
                    #rag2 = round(min1+((max1-min1)*0.9),3)
                    step = round((max1-min1)/20.0,3)
                    value = st.sidebar.slider(column, min1, max1, (min1,max1), step)
                    name2.append(column)
                    test2.append(value)
                    
                st.write('')
                st.write('')        
                st.write('')        
                st.write('**3.2 ?????? ????????????(CTP) ??????**')
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.2 ?????? ????????? CTQ ????????? ?????? ?????? ????????????(CTP)??? ???????????????. </h6>", unsafe_allow_html=True)
                
                if st.sidebar.button('CTP ????????????',key = count): 
                
                    para = []
                    para2 = []
                    para4 = []
                
                    #st.write(test2)
                    import itertools
                
                    for para in test2:
                        if para[0] == para[1]:
                            para = itertools.repeat(para[0],100)
                        else:
                            para = np.arange(round(para[0],6), round(para[1]+((para[1]-para[0])/100.0),6), round((para[1]-para[0])/100.0,6))
                    #st.write(para)
                        para2.append(para)
                
               
                    Iter = N_sample
                
                    para2 = pd.DataFrame(para2)
                    para2 = para2.T
                    para2 = para2.dropna().reset_index()
                    para2.drop(['index'],axis=1,inplace=True)
                
                    Iter2 = para2.shape[1]
                
                
    
                #st.write(Iter,Iter2)
                    for i in range(Iter):
                        para3 = []
                        para5 = []
                        for j in range(Iter2):
                            #st.write(i,j,list(para2[j]))
                            para3.append(random.sample(list(para2[j]),1))
                            para5.append(para3[j][0])
                        
                    #para3 = pd.DataFrame(para3).values
                    #para4.append(para3)
                        para4.append(para5)
                    
                    
                #para4 = pd.DataFrame(para4)
                    para4 = pd.DataFrame(para4)
                
                
                    para4.columns=list(Selected_X)
                    st.write('')
    
                    st.write('**??? ????????? ????????????(CTP) ?????? :**')
    
                    st.write(para4)
                    
                    datafile = para4
              
                    predictions2 = model.predict(datafile)
                    
                    predictions2 = pd.DataFrame(predictions2,columns=df[Selected_y].columns)
                
                    
                    para4 = pd.concat([para4, predictions2],axis=1)
                        
    
                
                    #para4.sort_values(by='predicted results', ascending=True, inplace =True)
                
                    para4 = para4.reset_index()
                    para4.drop(['index'],axis=1,inplace=True)
                
                    predictions3 = scaler_y.transform(predictions2)                
                    
                    Target = pd.DataFrame(Target)
                    Target = Target.T
                    Target2 = scaler_y.transform(Target)
                    
                    Target2 = pd.DataFrame(Target2)
                    Target3 = []
                    
                    for i in range(Iter):
                        Target3.append(Target2.values)
                    
                    Target3 = np.reshape(Target3, (Iter, Selected_y.shape[0]))
                    Target3 = pd.DataFrame(Target3)
                    
                    Diff3 = abs(predictions3 - Target3)
    
                    #################????????????################
                    Diff4 = Diff3 * Diff3
                    Diff5 = Diff4.sum(axis=1)
                    #Diff5.sort_values(ascending=True, inplace =True)
                    #para4['sum'] = Diff5.sum(axis=1)                   
                    para4['sum'] = Diff5
                    para4.sort_values(by='sum', ascending=True, inplace =True)
    
    
                   
                    """
                    def find_nearest(array, value):
                        
                        array = np.asarray(array)
                        idx = (np.abs(array - value)).argmin()
                        return array[idx]
                
                    opt = predictions2.shape[1]
                    
                    opt_result = []
                    for i in range(opt):
                        
                        opt_result.append(find_nearest(predictions2.iloc[:,i],Target[i]))
      
    
                    st.write(opt_result)
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * Optimizing Condition Results </h6>", unsafe_allow_html=True)
                
                
                    #df_max = para4[para4['predicted results']==para4['predicted results'].max()]
                    #df_min = para4[para4['predicted results']==para4['predicted results'].min()]
                    df_opt = pd.DataFrame()
                    
                    for i in range(opt):   
                        for column in df[Selected_y].columns:
                            df_opt = pd.concat([df_opt, para4[para4[column]==opt_result[i]]],axis=0)
                    """        
                    
                    st.write('**??? ?????? ????????????(CTP) :**')
                    opt = para4.drop(['sum'],axis=1)
                    st.write(opt.head(5))
                    
    
                    st.write('')
    
                    st.write('**??? ?????? ?????? ?????? ?????? :**')
                    st.write(opt)
                    
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    
                    Target5 = []
                    for i in range(Iter):
                        Target5.append(Target.values)
                    Target5 = np.reshape(Target5, (Iter, Selected_y.shape[0]))
                    Target5 = pd.DataFrame(Target5)
                    
                    plt.figure(figsize=(10,6))
                    fig, axs = plt.subplots(ncols=df[Selected_y].shape[1])
                    fig.subplots_adjust(hspace=1)
                
                    opt_best5 = opt.head(5)
                    opt_ctq_best5 = opt_best5[Selected_y]
                    
                    for i in range(1,df[Selected_y].shape[1]+1):
                        
                        plt.subplot(1,df[Selected_y].shape[1],i)
                        
                        sns.lineplot(x=para4.index, y=Target5[i-1], color='red')
                        
                        plt.scatter(para4.index, predictions2.iloc[:,i-1], color='#0e4194', label = 'Prediction')
                        plt.scatter(opt_ctq_best5.index, opt_ctq_best5.iloc[:,i-1], color='#e30613', label = 'Prediction')
                        
                        plt.title(df[Selected_y].columns[i-1],fontsize=10)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                    #ax.set_xlabel('Time', fontsize=16)
                    #plt.ylabel(Y.columns[i-1], fontsize=10)
                    
                    st.pyplot()
                    
    
                    count +=1
    
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown('**??? ?????? ?????? ????????????**')
        
                    st_pandas_to_csv_download_link(para4, file_name = "Predicted_CTPs_Results.csv")
                    st.caption("**_??? ?????? ?????? ?????? ?????? : ????????? ????????? ??? [?????? ???????????? ?????? ??????]_**")
                    
                    st.write('')
                    st.write('')
           
        
        
                st.sidebar.write('')
                st.sidebar.write('')    
                st.sidebar.write('')
                st.sidebar.write('**3.3 ????????????(CTQ) ?????? (Mulit-case)**')
                uploaded_file3 = st.sidebar.file_uploader("Mulit-case ?????????(.csv) ?????????", type=["csv"])
                if uploaded_file3 is not None:
                    def load_csv():
                        csv = pd.read_csv(uploaded_file3)
                        return csv
                
    
    
    #    uploaded_file = st.file_uploader("Choose a file")
    #if uploaded_file is not None:
    #    uploaded_file.seek(0)
    #    data = pd.read_csv(uploaded_file, low_memory=False)
    #    st.write(data.shape)
        
                count +=1
                st.write('')
                st.write('')        
                st.write('')        
                st.write('**3.3 ????????????(CTQ) ?????? (Multi-case)**')
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.3 ?????? ???????????? ????????? ???????????? ?????????(CTP)??? ?????? ?????? ??????(CTQ)??? ???????????????. </h6>", unsafe_allow_html=True)
                
                if st.sidebar.button('CTQ ????????????', key = count):
                    df3 = load_csv()            
                    datafile = df3
    
                    
                    predictions2 = model.predict(datafile)
    
                    predictions2 = pd.DataFrame(predictions2,columns=df[Selected_y].columns)
    
                    #df3['predicted results'] = predictions2
                    
                    predictions3 = pd.DataFrame()
                    predictions3 = pd.concat([df3, predictions2],axis=1)
                    
                    st.write("")
                    st.write('**??? ?????? ?????? ?????? ?????? :**')
                    st.write(predictions3)
    
    
                    
                
                    plt.figure(figsize=(10,6))
                    fig, axs = plt.subplots(ncols=df[Selected_y].shape[1])
                    fig.subplots_adjust(hspace=1)
                
                    for i in range(1,df[Selected_y].shape[1]+1):
                
    
                    
                        plt.subplot(1,df[Selected_y].shape[1],i)
                        
                        #sns.lineplot(x=df3.index,y=Target[i-1], color='red')
                        
                        #plt.plot(df[Selected_y].iloc[:,i-1], df[Selected_y].iloc[:,i-1], color='blue', label = 'Actual data')
                        plt.scatter(df3.index, predictions2.iloc[:,i-1], color='#e30613', label = 'Prediction')
                        plt.title(df[Selected_y].columns[i-1],fontsize=10)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                    #ax.set_xlabel('Time', fontsize=16)
                    #plt.ylabel(Y.columns[i-1], fontsize=10)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    
                
                    #st.write('')
                    #st.write('**Total results:**')
                    #st.set_option('deprecation.showPyplotGlobalUse', False)
                    #st.write(df3)
                    #sns.scatterplot(x=df3.index,y=df3.iloc[:,-1],s=30,color='red')
                    #st.pyplot()
                    count +=1
        
    
    
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown('**??? ?????? ?????? ???????????? (Multi-case)**')
            
                    st_pandas_to_csv_download_link(df3, file_name = "Predicted_CTQs_Results.csv")
                    st.caption("**_??? ?????? ?????? ?????? ?????? : ????????? ????????? ??? [?????? ???????????? ?????? ??????]_**")
                        
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
    
    
                    
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")


################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################


        # Main panel        
    if howtofetch == "Option2) ?????? ?????? ?????? ?????????":
        #---------------------------------#
        # Sidebar - Collects user input features into dataframe
        with st.sidebar.header('1. ?????? ?????????(.xlsx) ?????????'):
            uploaded_file = st.sidebar.file_uploader( 'Train Data File', type=["xlsx"])
        
        if uploaded_file is not None:
            def load_dataset():
                file = pd.read_excel(uploaded_file, sheet_name = 'sheet1')
                return file
            def load_num_y():
                y = pd.read_excel(uploaded_file, sheet_name = 'sheet2')
                return y
            
            st.subheader('**1. ???????????? ?????? ?????????**')
            st.write('')
            df = load_dataset()
            num_y = load_num_y()
            num_y = num_y.iloc[0,0]
            #st.write(num_y)
            
            x = list(df.columns)

            Selected_X = st.sidebar.multiselect('????????????(CTP) ????????????', x, x[:-num_y])
            
            y = [a for a in x if a not in Selected_X]
            
            Selected_y = st.sidebar.multiselect('????????????(CTQ) ????????????', y, y)

            Selected_X = np.array(Selected_X)
            Selected_y = np.array(Selected_y)
            
             
            st.write('**1.1 CTP ?????? :**',Selected_X.shape[0])
            st.info(list(Selected_X))
            st.write('')
        
            st.write('**1.2 CTQ ??????:**',Selected_y.shape[0])
            st.info(list(Selected_y))
        
            df2 = pd.concat([df[Selected_X],df[Selected_y]], axis=1)
            #df2 = df[df.columns == Selected_X]
        
            #Selected_xy = np.array((Selected_X,Selected_y))
            
            st.write('')
            st.write('')
            st.write('')
            st.write('')    
            st.write('')    
            
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.write("")
            with st.sidebar.header('2. ????????????(.pkl) ?????????'):
                uploaded_file2 = st.sidebar.file_uploader("Trained model file", type=["pkl"])
                
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.write("")
                
            
            if uploaded_file2 is not None:
                def load_model(model):
                    loaded_model = pickle.load(model)
                    return loaded_model
                
                st.subheader('**2. ???????????? ???????????? ??????**')
                st.write('')

                model = load_model(uploaded_file2)
                

                
                st.markdown('**2.1 ????????? ???????????? ?????? :**')
                st.write(model)
                st.write('')
                
                st.write('')
                st.write('')
                st.markdown('**2.2 ?????? ?????? ????????? :**')
                X_train = df[Selected_X]
                y_train = df[Selected_y]
                
                scaler_y = StandardScaler().fit(y_train)
        
                #rescaled = scaler.transform(X_train)
                
                predictions = model.predict(X_train)
            
      
            
                results = []

                msg = []
                mean = []
                std = []        

                    
                kfold = KFold(n_splits=5, random_state=7, shuffle=True)
                cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
                
                for i, element in enumerate(cv_results):
                    if element <= 0.0:
                        cv_results[i] = 0.0
                        
                        
                results.append(cv_results)
                #    names.append(name)
                msg.append('%s' % model)
                mean.append('%f' %  (cv_results.mean()))
                std.append('%f' % (cv_results.std()))
                        
                        
                F_result3 = pd.DataFrame(np.transpose(msg))
                F_result3.columns = ['Machine_Learning_Model']
                F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
                
                #st.write(F_result3)    

                st.write('?????? ????????? ($R^2$):')
                
                R2_mean = list(F_result3['R2_Mean'].values)
                st.info( R2_mean[0] )
                    
                st.write('?????? ????????? ?????? (Standard Deviation):')
                
                R2_std = list(F_result3['R2_Std'].values)
                st.info( R2_std[0])
                    
                
                
                
                            
                
                df2 = df[Selected_X]
                columns = df2.columns
        
                test = []
                name = []
                #st.sidebar.write('3.1 Predict Single Condition')
                

    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
        
        
                if Selected_y.shape[0] <= 1:
                    st.sidebar.header('3. CTP, CTQ ??????')
                    st.sidebar.write('**3.1 ????????????(CTQ) ??????**')
                    
                    #st.subheader('**3. Model Prediction **')
                    #st.write('**3.1 Single Condition Prediction :**')

                    for column in columns:
                            value = st.sidebar.number_input(column, None, None, df2[column].mean(),format="%.2f") #int(df2[column].mean()))
                    
                            name.append(column)
                            test.append(value)
                    
     

                    
                    st.write('')
                    st.write('')
                    st.write('')        
                    st.write('')        
                    st.write('')        
                                  
                    st.subheader('**3. CTP, CTQ ??????**')
         
                    st.write('')        
                    st.write('**3.1 ????????????(CTQ) ?????? - ?????? ?????????**')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.1 ?????? ????????? ????????????(CTP)??? ?????? ?????? ??????(CTQ)??? ???????????????. </h6>", unsafe_allow_html=True)
                    st.write('')        

                    
                    F_result = pd.DataFrame(name)
                    F_result.columns = ['X Variables']
                    F_result['Value'] = pd.DataFrame(test)

                    
                    #
         
                #para = st.sidebar.slider(para,mi,ma,cu,inter)
                    #st.write('')
                
                
            
                    #st.write(F_result)
                    #scaler = StandardScaler().fit(X_train)
                
                    if st.sidebar.button('CTQ ????????????'):
                
                #st.write(F_result)
                        F_result = F_result.set_index('X Variables')
                        F_result_T = F_result.T
        
                        #rescaled2 = scaler.transform(F_result_T)
            
                        predictions = model.predict(F_result_T)
            
                        st.markdown("_- ????????? ????????????(CTP) ??????_")
                        st.write(F_result)                    
        
                        st.write('**??? CTQ [', Selected_y[0],"]??? ?????? ?????????  :**" , predictions[0])
                        
                    
     
                    
                    
                    
                    st.sidebar.write('')
                    st.sidebar.write('')    
                    st.sidebar.write('**3.2 ?????? ????????????(CTP) ??????**')
                    
                    ctq1 = df[Selected_y].mean()
                    ctq2 = ctq1[0]

                    
                #Target = st.sidebar.number_input(list(Selected_y)[0], 0.00, 1000.00, 0.00,format="%.2f")
                    st.sidebar.write('**Step1) CTQ ?????? ??????????**')
                    N_sample = 0
                    Target = st.sidebar.number_input(Selected_y[0], None, None, ctq2,format="%.2f")
                
                    st.sidebar.write("")    
                    st.sidebar.write('**Step2) ????????? ?????? ??????????**')
                    N_sample = st.sidebar.number_input("?????? ??????",0, 1000, 50, format="%d")
                    
                    name2=[]
                    test2=[]
                    count = 0
                    
                    st.sidebar.write("")    
                    st.sidebar.write('**Step3) CTP ?????? ????????????**')
                    
                    for column in columns:
                    
                        max1 = round(float(df[column].max()),3)
                        min1 = round(float(df[column].min()),3)
                    
                        #rag1 = round(min1+((max1-min1)*0.1),3)
                        #rag2 = round(min1+((max1-min1)*0.9),3)
                    
                        step = round((max1-min1)/20.0,3)
                    
                        value = st.sidebar.slider(column, min1, max1, (min1,max1), step)
                             
                        name2.append(column)
                        test2.append(value)
                    #param2.append(para_range)
                    #st.write(min1,rag1,rag2,max1)
                    #st.write(column)
                    #st.write(test2)
                    st.write('')        
                    st.write('')        
                    st.write('')
                    st.write('**3.2 ?????? ????????????(CTP) ??????**')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.2 ?????? ????????? CTQ ????????? ?????? ?????? ????????????(CTP)??? ???????????????. </h6>", unsafe_allow_html=True)
            
                    if st.sidebar.button('CTP ????????????',key = count): 
                    
                        para = []
                        para2 = []
                        para4 = []
                    
                        #st.write(test2)
                        import itertools
                    
                        for para in test2:
                            if para[0] == para[1]:
                                para = itertools.repeat(para[0],100)
                            else:
                                para = np.arange(round(para[0],6), round(para[1]+((para[1]-para[0])/100.0),6), round((para[1]-para[0])/100.0,6))
                        #st.write(para)
                            para2.append(para)
                    
                   
                        Iter = N_sample
                    
                        para2 = pd.DataFrame(para2)
                        para2 = para2.T
                        para2 = para2.dropna().reset_index()
                        para2.drop(['index'],axis=1,inplace=True)
                    
                        Iter2 = para2.shape[1]
                    
                    

                    #st.write(Iter,Iter2)
                        for i in range(Iter):
                            para3 = []
                            para5 = []
                            for j in range(Iter2):
                                #st.write(i,j,list(para2[j]))
                                para3.append(random.sample(list(para2[j]),1))
                                para5.append(para3[j][0])
                            
                        #para3 = pd.DataFrame(para3).values
                        #para4.append(para3)
                            para4.append(para5)
                        
                        
                    #para4 = pd.DataFrame(para4)
                        para4 = pd.DataFrame(para4)
                    
                    
                        para4.columns=list(Selected_X)
                    
                        #st.write('_????????? ?????? ??????????????? ?????? :_')
                        #st.write(para4)
                    
                        datafile = para4
        
                        #rescaled = scaler.transform(datafile)
                
                        predictions2 = model.predict(datafile)
        
                        para4['predicted results'] = predictions2
                    
                    #st.write(para4)
                    
                        para4.sort_values(by='predicted results', ascending=True, inplace =True)
                    
                        para4 = para4.reset_index()
                        para4.drop(['index'],axis=1,inplace=True)
                    
                    #st.write(para4)
                    
                        def find_nearest(array, value):
                            array = np.asarray(array)
                            idx = (np.abs(array - value)).argmin()
                            return array[idx]
                    
                        opt_result = find_nearest(para4['predicted results'],Target)
                    
                    
                    #st.write(opt_result)
        
                        st.write('')
                        st.markdown("_- ????????? ?????? ?????? : %.2f_" %Target)
                        st.markdown("_- ????????? ?????? ?????? : %d_" %N_sample)
                        st.write('')               
                    
                        df_max = para4[para4['predicted results']==para4['predicted results'].max()]
                        df_min = para4[para4['predicted results']==para4['predicted results'].min()]
                        df_opt = para4[para4['predicted results']==opt_result]
                        st.write('**??? ?????? ????????????(CTP) :**')
                        st.write(df_opt)
                        st.write("")
                        st.write('**??? ?????? CTQ ???????????? :**')
                        st.write(df_max)
                        st.write("")
                        st.write('**??? ?????? CTQ ???????????? :**')
                        st.write(df_min)
                        #st.info(list(Selected_X2))
                        st.write("")
                        st.write('**??? ?????? ?????? ?????? ?????? :**')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.write(para4)
                        st.write('')
             
                        #sample_list_1 = para4
                        #del sample_list_1["predicted results"]
                        #st.write(sample_list_1)


                        opt_ctq_best1 = df_opt['predicted results']

               
                        '''
                        ax = sns.scatterplot(x=para4.index,y=para4.iloc[:,-1],s=30,color='#0e4194')
                        #ax1 = sns.scatterplot(x=opt_ctq_best1.index,y=opt_ctq_best1.iloc[:,-1],s=30,color='#e30613')
                        sns.lineplot(x=para4.index,y=Target,ax=ax.axes, color='red')
                        '''
                        plt.scatter(para4.index, para4.iloc[:,-1],color='#0e4194')
                        plt.scatter(opt_ctq_best1.index, opt_ctq_best1.iloc[0], color='#e30613')
                        sns.lineplot(x=para4.index, y=Target, color='red')
                        
                        st.pyplot()
                    
                    
                        count +=1

                        st.write('')
                        st.write('')
                        st.write('')
                        st.markdown('**??? ?????? ?????? ????????????**')
            
                        st_pandas_to_csv_download_link(para4, file_name = "Predicted_CTPs_Results.csv")
                        st.caption("**_??? ?????? ?????? ?????? ?????? : ????????? ????????? ??? [?????? ???????????? ?????? ??????]_**")
                        st.write('')
                        st.write('')


                
            
            
                    st.sidebar.write('')
                    st.sidebar.write('')    
                
            
                    st.sidebar.write("**3.3 ????????????(CTQ) ?????? (Multi-case)**")
                    uploaded_file3 = st.sidebar.file_uploader("Multi-case ?????????(.csv) ?????????", type=["csv"])
                    if uploaded_file3 is not None:
                        def load_csv():
                            csv = pd.read_csv(uploaded_file3)
                            return csv
                    
        
        

            
                    count +=1
                    st.write('')        
                    st.write('')        
                    st.write('')        
                    st.write('**3.3 ????????????(CTQ) ?????? (Multi-case)**')
                    
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.3 ?????? ???????????? ????????? ???????????? ?????????(CTP)??? ?????? ?????? ??????(CTQ)??? ???????????????. </h6>", unsafe_allow_html=True)
                
                    if st.sidebar.button('CTQ ????????????',key = count):
                        df3 = load_csv()            
                        datafile = df3
        
                        #rescaled = scaler.transform(datafile)
                        
                        predictions2 = model.predict(datafile)
        
                        df3['predicted results'] = predictions2
        
                        
                    
                        df_max = df3[df3['predicted results']==df3['predicted results'].max()]
                        df_min = df3[df3['predicted results']==df3['predicted results'].min()]
                        st.write("")
                        st.write('**??? ?????? CTQ ???????????? :**')
                        st.write(df_max)
                        st.write('**??? ?????? CTQ ???????????? :**')
                        st.write(df_min)
                        #st.info(list(Selected_X2))
                    
                        st.write('')
                        st.write('**??? ?????? ?????? ?????? ?????? :**')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.write(df3)
                        sns.scatterplot(x=df3.index,y=df3.iloc[:,-1],s=30,color='#e30613')
                        st.write('')
                        st.pyplot()
                        count +=1
            
                        st.write('')
                        st.write('')
                        st.write('')
                        st.markdown('**??? ?????? ?????? ???????????? (Multi-case)**')
            
                        st_pandas_to_csv_download_link(df3, file_name = "Predicted_CTQs_Results.csv")
                        st.caption("**_??? ?????? ?????? ?????? ?????? : ????????? ????????? ??? [?????? ???????????? ?????? ??????]_**")
                        
                        st.write('')
                        st.write('')
                        st.write('')
                        st.write('')
                        st.write('')


                    
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")

                #st.write(mi)
            #parameter_n_neighbors = st.sidebar.slider('Number of neighbers', 2, 10, (1,6), 2)  

            
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################


            
                else :
                    
                    df_y = df[Selected_y]
                    
                    st.sidebar.header('3. CTP, CTQ ??????')
                    st.sidebar.write('**3.1 ????????????(CTQ) ??????**')
                    
                    #st.subheader('**3. Model Prediction **')
                    #st.write('**3.1 Single Condition Prediction :**')

                    for column in columns:
                            value = st.sidebar.number_input(column, None, None, df2[column].mean(),format="%.2f") #int(df2[column].mean()))
                    
                            name.append(column)
                            test.append(value)
                    
     

                    
                    st.write('')
                    st.write('')
                    st.write('')        
                    st.write('')        
                    st.write('')        
                                  
                    st.subheader('**3. CTP, CTQ ??????**')
                    st.write('')        
                    st.write('**3.1 ????????????(CTQ) ?????? - ?????? ?????????**')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.1 ?????? ????????? ????????????(CTP)??? ?????? ?????? ??????(CTQ)??? ???????????????. </h6>", unsafe_allow_html=True)
                    F_result = pd.DataFrame(name)
                    F_result.columns = ['X Variables']
                    F_result['Value'] = pd.DataFrame(test)
                    
                    
                    
                    #st.write(F_result)
         
                #para = st.sidebar.slider(para,mi,ma,cu,inter)
                    st.write('')
                
                
            
                    
                    
                
                    if st.sidebar.button('CTQ ????????????'):
                
                #st.write(F_result)
                        F_result = F_result.set_index('X Variables')
                        F_result_T = F_result.T
        
          
                        predictions = model.predict(F_result_T)
            

                        #predictions = pd.DataFrame(predictions[0],columns=['Value'])
                        #st.write(predictions)
                        
                        predictions2 = pd.DataFrame()
                        predictions2['Y Variable'] = df[Selected_y].columns
                        predictions2['Value'] = pd.DataFrame(predictions[0])
                        
                        
                        #Selected_y_list = Selected_y.tolist()
                        #st.write(Selected_y_list)
                        
                        st.write("_????????? ????????????(CTP) ??????_")
                        st.write(F_result)
                        
                        st.write("**??? CTQ(Y variables)??? ?????? ????????? :**")
                        st.write(predictions2)
                        
                        
                    
                    
                    
                    st.sidebar.write('')
                    st.sidebar.write('')    
                    st.sidebar.write('**3.2 ?????? ????????????(CTP) ??????**')
                
                #Target = st.sidebar.number_input(list(Selected_y)[0], 0.00, 1000.00, 0.00,format="%.2f")
                    st.sidebar.write('**Step1) CTQ ?????? ??????????**')
                    
                    sample_target = df[Selected_y].shape[1]
                                    
                    Target = []
                    

                    for i in range(sample_target):

                        Sel_y_mean = df[Selected_y[i]]
                        Sel_y_mean2 = Sel_y_mean.mean()
     
                        Target.append(st.sidebar.number_input(Selected_y[i], None, None, Sel_y_mean2, format="%.2f"))
                
                
                
                    N_sample = 0
                    st.sidebar.write('')
                    st.sidebar.write('**Step2) ????????? ?????? ??????????**')
                    N_sample = st.sidebar.number_input('?????? ??????',0,1000,50, format="%d")
                    
                
                    
                    name2=[]
                    test2=[]
                    count = 0
                    
                    st.sidebar.write('')
                    st.sidebar.write('**Step3) CTP ?????? ????????????**')
                    for column in columns:
                    
                        max1 = round(float(df[column].max()),3)
                        min1 = round(float(df[column].min()),3)
                        
                        
                        #rag1 = round(min1+((max1-min1)*0.1),3)
                        #rag2 = round(min1+((max1-min1)*0.9),3)
                        step = round((max1-min1)/20.0,3)
                        value = st.sidebar.slider(column, min1, max1, (min1,max1), step)
                        name2.append(column)
                        test2.append(value)
                        
                    st.write('')
                    st.write('')        
                    st.write('')        
                    st.write('**3.2 ?????? ????????????(CTP) ??????**')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.2 ?????? ????????? CTQ ????????? ?????? ?????? ????????????(CTP)??? ???????????????. </h6>", unsafe_allow_html=True)
                    
                    if st.sidebar.button('CTP ????????????',key = count): 
                    
                        para = []
                        para2 = []
                        para4 = []
                    
                        #st.write(test2)
                        import itertools
                    
                        for para in test2:
                            if para[0] == para[1]:
                                para = itertools.repeat(para[0],100)
                            else:
                                para = np.arange(round(para[0],6), round(para[1]+((para[1]-para[0])/100.0),6), round((para[1]-para[0])/100.0,6))
                        #st.write(para)
                            para2.append(para)
                    
                   
                        Iter = N_sample
                    
                        para2 = pd.DataFrame(para2)
                        para2 = para2.T
                        para2 = para2.dropna().reset_index()
                        para2.drop(['index'],axis=1,inplace=True)
                    
                        Iter2 = para2.shape[1]
                    
                    

                    #st.write(Iter,Iter2)
                        for i in range(Iter):
                            para3 = []
                            para5 = []
                            for j in range(Iter2):
                                #st.write(i,j,list(para2[j]))
                                para3.append(random.sample(list(para2[j]),1))
                                para5.append(para3[j][0])
                            
                        #para3 = pd.DataFrame(para3).values
                        #para4.append(para3)
                            para4.append(para5)
                        
                        
                    #para4 = pd.DataFrame(para4)
                        para4 = pd.DataFrame(para4)
                    
                    
                        para4.columns=list(Selected_X)
                        st.write('')

                        st.write('**??? ????????? ????????????(CTP) ?????? :**')

                        st.write(para4)
                        
                        datafile = para4
                  
                        predictions2 = model.predict(datafile)
                        
                        predictions2 = pd.DataFrame(predictions2,columns=df[Selected_y].columns)
                    
                        
                        para4 = pd.concat([para4, predictions2],axis=1)
                            

                    
                        #para4.sort_values(by='predicted results', ascending=True, inplace =True)
                    
                        para4 = para4.reset_index()
                        para4.drop(['index'],axis=1,inplace=True)
                    
                        predictions3 = scaler_y.transform(predictions2)                
                        
                        Target = pd.DataFrame(Target)
                        Target = Target.T
                        Target2 = scaler_y.transform(Target)
                        
                        Target2 = pd.DataFrame(Target2)
                        Target3 = []
                        
                        for i in range(Iter):
                            Target3.append(Target2.values)
                        
                        Target3 = np.reshape(Target3, (Iter, Selected_y.shape[0]))
                        Target3 = pd.DataFrame(Target3)
                        
                        Diff3 = abs(predictions3 - Target3)

                        #################????????????################
                        Diff4 = Diff3 * Diff3
                        Diff5 = Diff4.sum(axis=1)
                        #Diff5.sort_values(ascending=True, inplace =True)
                        #para4['sum'] = Diff5.sum(axis=1)                   
                        para4['sum'] = Diff5
                        para4.sort_values(by='sum', ascending=True, inplace =True)


                       
                        """
                        def find_nearest(array, value):
                            
                            array = np.asarray(array)
                            idx = (np.abs(array - value)).argmin()
                            return array[idx]
                    
                        opt = predictions2.shape[1]
                        
                        opt_result = []
                        for i in range(opt):
                            
                            opt_result.append(find_nearest(predictions2.iloc[:,i],Target[i]))
      
        
                        st.write(opt_result)
                        st.write('')
                        st.write('')
                        st.write('')
                        st.markdown("<h6 style='text-align: left; color: #0e4194;'> * Optimizing Condition Results </h6>", unsafe_allow_html=True)
                    
                    
                        #df_max = para4[para4['predicted results']==para4['predicted results'].max()]
                        #df_min = para4[para4['predicted results']==para4['predicted results'].min()]
                        df_opt = pd.DataFrame()
                        
                        for i in range(opt):   
                            for column in df[Selected_y].columns:
                                df_opt = pd.concat([df_opt, para4[para4[column]==opt_result[i]]],axis=0)
                        """        
                        
                        st.write('**??? ?????? ????????????(CTP) :**')
                        opt = para4.drop(['sum'],axis=1)
                        st.write(opt.head(5))
                        

                        st.write('')

                        st.write('**??? ?????? ?????? ?????? ?????? :**')
                        st.write(opt)
                        
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        
                        Target5 = []
                        for i in range(Iter):
                            Target5.append(Target.values)
                        Target5 = np.reshape(Target5, (Iter, Selected_y.shape[0]))
                        Target5 = pd.DataFrame(Target5)
                        
                        plt.figure(figsize=(10,6))
                        fig, axs = plt.subplots(ncols=df[Selected_y].shape[1])
                        fig.subplots_adjust(hspace=1)
                    
                        opt_best5 = opt.head(5)
                        opt_ctq_best5 = opt_best5[Selected_y]
                        
                        for i in range(1,df[Selected_y].shape[1]+1):
                            
                            plt.subplot(1,df[Selected_y].shape[1],i)
                            
                            sns.lineplot(x=para4.index, y=Target5[i-1], color='red')
                            
                            plt.scatter(para4.index, predictions2.iloc[:,i-1], color='#0e4194', label = 'Prediction')
                            plt.scatter(opt_ctq_best5.index, opt_ctq_best5.iloc[:,i-1], color='#e30613', label = 'Prediction')
                            
                            plt.title(df[Selected_y].columns[i-1],fontsize=10)
                            plt.xticks(fontsize=8)
                            plt.yticks(fontsize=8)
                        #ax.set_xlabel('Time', fontsize=16)
                        #plt.ylabel(Y.columns[i-1], fontsize=10)
                        
                        st.pyplot()
                        

                        count +=1

                        st.write('')
                        st.write('')
                        st.write('')
                        st.markdown('**??? ?????? ?????? ????????????**')
            
                        st_pandas_to_csv_download_link(para4, file_name = "Predicted_CTPs_Results.csv")
                        st.caption("**_??? ?????? ?????? ?????? ?????? : ????????? ????????? ??? [?????? ???????????? ?????? ??????]_**")
                        
                        st.write('')
                        st.write('')
               
            
            
                    st.sidebar.write('')
                    st.sidebar.write('')    
                    st.sidebar.write('')
                    st.sidebar.write('**3.3 ????????????(CTQ) ?????? (Mulit-case)**')
                    uploaded_file3 = st.sidebar.file_uploader("Mulit-case ?????????(.csv) ?????????", type=["csv"])
                    if uploaded_file3 is not None:
                        def load_csv():
                            csv = pd.read_csv(uploaded_file3)
                            return csv
                    
        
        
    #    uploaded_file = st.file_uploader("Choose a file")
    #if uploaded_file is not None:
    #    uploaded_file.seek(0)
    #    data = pd.read_csv(uploaded_file, low_memory=False)
    #    st.write(data.shape)
            
                    count +=1
                    st.write('')
                    st.write('')        
                    st.write('')        
                    st.write('**3.3 ????????????(CTQ) ?????? (Multi-case)**')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * ????????????_3.3 ?????? ???????????? ????????? ???????????? ?????????(CTP)??? ?????? ?????? ??????(CTQ)??? ???????????????. </h6>", unsafe_allow_html=True)
                    
                    if st.sidebar.button('CTQ ????????????', key = count):
                        df3 = load_csv()            
                        datafile = df3
        
                        
                        predictions2 = model.predict(datafile)
        
                        predictions2 = pd.DataFrame(predictions2,columns=df[Selected_y].columns)
        
                        #df3['predicted results'] = predictions2
                        
                        predictions3 = pd.DataFrame()
                        predictions3 = pd.concat([df3, predictions2],axis=1)
                        
                        st.write("")
                        st.write('**??? ?????? ?????? ?????? ?????? :**')
                        st.write(predictions3)
        
        
                        
                    
                        plt.figure(figsize=(10,6))
                        fig, axs = plt.subplots(ncols=df[Selected_y].shape[1])
                        fig.subplots_adjust(hspace=1)
                    
                        for i in range(1,df[Selected_y].shape[1]+1):
                    

                        
                            plt.subplot(1,df[Selected_y].shape[1],i)
                            
                            #sns.lineplot(x=df3.index,y=Target[i-1], color='red')
                            
                            #plt.plot(df[Selected_y].iloc[:,i-1], df[Selected_y].iloc[:,i-1], color='blue', label = 'Actual data')
                            plt.scatter(df3.index, predictions2.iloc[:,i-1], color='#e30613', label = 'Prediction')
                            plt.title(df[Selected_y].columns[i-1],fontsize=10)
                            plt.xticks(fontsize=8)
                            plt.yticks(fontsize=8)
                        #ax.set_xlabel('Time', fontsize=16)
                        #plt.ylabel(Y.columns[i-1], fontsize=10)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                        
                    
                        #st.write('')
                        #st.write('**Total results:**')
                        #st.set_option('deprecation.showPyplotGlobalUse', False)
                        #st.write(df3)
                        #sns.scatterplot(x=df3.index,y=df3.iloc[:,-1],s=30,color='red')
                        #st.pyplot()
                        count +=1
            
        

                        st.write('')
                        st.write('')
                        st.write('')
                        st.markdown('**??? ?????? ?????? ???????????? (Multi-case)**')
                
                        st_pandas_to_csv_download_link(df3, file_name = "Predicted_CTQs_Results.csv")
                        st.caption("**_??? ?????? ?????? ?????? ?????? : ????????? ????????? ??? [?????? ???????????? ?????? ??????]_**")
                            
                        st.write('')
                        st.write('')
                        st.write('')
                        st.write('')
                        st.write('')


                        
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")

                #st.write(mi)
            #parameter_n_neighbors = st.sidebar.slider('Number of neighbers', 2, 10, (1,6), 2)  
      

