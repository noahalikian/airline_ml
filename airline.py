import pandas as pd
import numpy as np
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def file_read(file):
    return pd.read_csv(str(file))

default = file_read('airline.csv')

@st.cache_resource
def pickle_load(pickle_file):
    dt_class = open(str(pickle_file), 'rb')
    x = pickle.load(dt_class)
    dt_class.close()
    return x

clf = pickle_load('airline_dt.pickle')

st.title('Airline Customer Satisfaction')
st.subheader('Gain insights into passenger experiences and improve satisfaction through data analysis and surveys')
st.image('airline.jpg')
st.caption('Understand your customers to improve your airline services!')
with st.expander('What can you do with this app?'):
    st.markdown('<b>Fill Out a Survey:</b> Provide a form for users to fill out their airline satisfaction feedback.', unsafe_allow_html=True)
    st.markdown('<b>Make Data-Driven Decisions:</b> use Insights to guide improvements in customer experience.', unsafe_allow_html=True)
    st.markdown('<b>Interactive Features:</b> Explore data with fully interactive charts and summaries!', unsafe_allow_html=True)
st.title('Prediction of Customer Satisfaction (Decision Tree)')

st.sidebar.header('Airline Customer Satisfaction Survey')
st.sidebar.subheader('Part 1: Customer Details')
st.sidebar.text('Provide information about the customer flying')
cust_type = st.sidebar.selectbox('What type of customer is this?', ['Loyal Customer', 'disloyal Customer'])
trav_reas = st.sidebar.selectbox('Is the customer traveling for business or personal reasons?', ['Personal Travel', 'Business travel'])
cust_class = st.sidebar.selectbox('In which class will the customer be flying', ['Eco', 'Eco Plus', 'Business'])
cust_age = st.sidebar.number_input('How old is the customer?')

st.sidebar.subheader('Part 2: Flight Details')
st.sidebar.text("Provide details about the customer's flight details")
fly_dist = st.sidebar.number_input('How far is the customer flying in miles?')
dep_mins_delayed = st.sidebar.number_input("How many minutes was the customer's departure delayed? (Enter 0 if not delayed)")
arr_mins_delayed = st.sidebar.number_input("How many minutes was the customer's arrival delayed? (Enter 0 if not delayed)")

st.sidebar.subheader('Part 3: Customer Experience')
st.sidebar.text("Provide details about the customer's flight experience and satisfaction")
seat_comf = st.sidebar.radio('How comfortable was the seat for the customer? (1-5 stars)', [1, 2, 3, 4, 5], horizontal=True)
dep_arr_conv = st.sidebar.radio("Was the departure/arrival time convenient for the customer? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
food_drink_rate = st.sidebar.radio("How would the customer rate the food and drink? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
gate_rate = st.sidebar.radio("How would the customer rate the gate location? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
wifi_rate = st.sidebar.radio("How would the customer rate the inflight wifi service? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
ent_rate = st.sidebar.radio("How would the customer rate the inflight entertainment? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
online_supp_rate = st.sidebar.radio("How would the customer rate online support? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
online_book = st.sidebar.radio("How easy was online booking for the customer? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
onboard_service_rate = st.sidebar.radio("How would the customer rate the onboard service? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
leg_room_rate = st.sidebar.radio("How would the customer rate the leg room service? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
baggage_rate = st.sidebar.radio("How would the customer rate baggage handling? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
check_in_rate = st.sidebar.radio("How would the customer rate the check-in service? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
clean_rate = st.sidebar.radio("How would the customer rate cleanliness? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
online_board_rate = st.sidebar.radio("How would the customer rate online boarding? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)

butt = st.sidebar.button('Predict')

if not butt:
    st.info('Please fill out the survey form in the sidebar and click Predict to see the satisfaction prediction.')

if butt:
    encoded = default.copy()
    encoded = encoded.drop(columns = ['satisfaction'])

    encoded.loc[len(encoded)] = [cust_type, cust_age, trav_reas, cust_class, fly_dist, seat_comf, dep_arr_conv, food_drink_rate, 
                                 gate_rate, wifi_rate, ent_rate, online_supp_rate, online_book, onboard_service_rate, leg_room_rate, 
                                 baggage_rate, check_in_rate, clean_rate, online_board_rate, dep_mins_delayed, arr_mins_delayed]
    
    dum_encoded = pd.get_dummies(encoded)

    dum_encoded_form_sub = dum_encoded.tail(1)

    new_pred = clf.predict(dum_encoded_form_sub)[0]
    probs = clf.predict_proba(dum_encoded_form_sub).max()

    st.markdown(f'# Prediction Result')
    st.markdown(f'The customer is **{new_pred}** with a confidence of **{probs*100}%**')

    st.subheader('Customer Demographic Analysis')
    default_cust_type = default[default['customer_type'] == cust_type]
    filter_count = default_cust_type['customer_type'].count()
    count = default['customer_type'].count()
    with st.expander('Customer Type Comparison'):
        st.markdown(f'**Customer Type:** Your selection - *{cust_type}*')
        st.markdown(f'Percentage of our fliers with this selection: **{(filter_count/count)*100:.2f}%**')
    
    default_trav_reas = default[default['type_of_travel'] == trav_reas]
    filter_count1 = default_trav_reas['type_of_travel'].count()
    count1 = default['type_of_travel'].count()
    with st.expander('Type of Travel Comparison'):
        st.markdown(f'**Customer Type:** Your selection - *{trav_reas}*')
        st.markdown(f'Percentage of our fliers with this selection: **{(filter_count1/count1)*100:.2f}%**')

    default_cust_class = default[default['class'] == cust_class]
    filter_count2 = default_cust_class['class'].count()
    count2 = default['class'].count()
    with st.expander('Flight Class Comparison'):
        st.markdown(f'**Customer Type:** Your selection - *{cust_class}*')
        st.markdown(f'Percentage of our fliers with this selection: **{(filter_count2/count2)*100:.2f}%**')
    
    eighteen_thirty = 0
    thirty_one_forty_five = 0
    forty_six_sixty = 0
    sixty_one_seventy_five = 0
    seventy_six_ninety = 0
    for i in default['age']:
        if i >=18 and i <= 30:
            eighteen_thirty += 1
        elif i >= 31 and i <= 45:
            thirty_one_forty_five +=1
        elif i >= 46 and i <= 60:
            forty_six_sixty += 1
        elif i >= 61 and i <= 75:
            sixty_one_seventy_five += 1
        elif i >= 76 and i <= 90:
            seventy_six_ninety += 1
    
    count3 = default['age'].count()
    with st.expander('Age Group Comparison'):
        if cust_age >=18 and cust_age <= 30:
            st.markdown(f'**Customer Type:** Your selection - *{cust_age} years old*')
            st.markdown('Your selected age group: *18-30*')
            st.markdown(f'Percentage of our fliers with this selection: **{(eighteen_thirty/count3)*100:.2f}%**')  
        elif cust_age >=31 and cust_age <= 45:
            st.markdown(f'**Customer Type:** Your selection - *{cust_age} years old*')
            st.markdown('Your selected age group: *31-45*')
            st.markdown(f'Percentage of our fliers with this selection: **{(thirty_one_forty_five/count3)*100:.2f}%**') 
        elif cust_age >=46 and cust_age <= 60:
            st.markdown(f'**Customer Type:** Your selection - *{cust_age} years old*')
            st.markdown('Your selected age group: *46-60*')
            st.markdown(f'Percentage of our fliers with this selection: **{(forty_six_sixty/count3)*100:.2f}%**')
        elif cust_age >=61 and cust_age <= 75:
            st.markdown(f'**Customer Type:** Your selection - *{cust_age} years old*')
            st.markdown('Your selected age group: *61-75*')
            st.markdown(f'Percentage of our fliers with this selection: **{(sixty_one_seventy_five/count3)*100:.2f}%**') 
        elif cust_age >=76 and cust_age <= 90:
            st.markdown(f'**Customer Type:** Your selection - *{cust_age} years old*')
            st.markdown('Your selected age group: *76-90*')
            st.markdown(f'Percentage of our fliers with this selection: **{(seventy_six_ninety/count3)*100:.2f}%**')
        else:
            st.markdown('**Please input an age between *18 - 90* in the sidebar**')     