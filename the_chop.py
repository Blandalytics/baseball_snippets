import streamlit as st
st.set_page_config(page_title='Is "The Chop" Racist?', page_icon='⚾')
st.header('Is "The Chop" racist?') 
check_box = st.checkbox('Are you Indigenous/Native American?')
if check_box:
  st.write("It's your call.")
else:
  st.write("Yeah, [it is](https://www.cnn.com/2021/10/28/us/atlanta-braves-tomahawk-chop).")
