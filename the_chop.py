import streamlit as st
st.set_page_config(page_title='Is "The Chop" Racist?', page_icon='⚾')
st.header('Is "The Chop" racist?') 
choice = st.pills('Are you Indigenous/Native American?',['Yes','No'])
if choice == 'Yes':
  st.write("It's your call.")
elif choice == 'No':
  st.write("Yeah, [it is](https://www.cnn.com/2021/10/28/us/atlanta-braves-tomahawk-chop).")
