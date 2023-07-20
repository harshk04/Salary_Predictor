import pandas as pd
import streamlit as st



st.set_page_config(page_title="Salary Prediction",page_icon=":tada:",layout="wide")
st.header("Salary Prediction Model")
st.sidebar.header("Enter All Details")

exp=st.sidebar.slider ("No. of experience" ,0,15,7)
cgpa=st.sidebar.slider("CGPA Score" ,0, 10,5) 
age=st.sidebar. slider("Age", 18,60, 40) 
interview_score=st.sidebar.slider("Interview Score",0, 100, 50)
# exp=st.sidebar.number_input ("No. of experience" ,0,15,7)
# age=st.sidebar.number_input('Age')
# cgpa=st.sidebar.number_input("CGPA Score" ,0, 10,5) 
# interview_score=st.sidebar.number_input("Interview Score",0, 100, 50)

ds=pd.read_csv("/Users/LNB/ML/salary model.csv")
# print(ds)

x=ds.iloc[:,:4]
y=ds.iloc[:,4:]
# print(x)
# print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)



y_pred=regression.predict(x_test)

from sklearn.metrics import r2_score

r2=r2_score(y_test,y_pred)
r2=r2*100
r2 = round(r2, 2)

if st.sidebar.button('Predict Salary'):
    pred=regression.predict([[exp,cgpa,age,interview_score]])
    st.subheader("Predicted Salary (in ₹)")
    st.write(pred)
    st.write("Accuracy of Model is", r2,"%")

import streamlit as st


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/harsh-kumawat-069bb324b/" target="_blank">Harsh</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)