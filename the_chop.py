import streamlit as st
st.set_page_config(page_title='Is "The Chop" Racist?', page_icon='⚾')
st.header('Is "The Chop" racist?') 
choice = st.pills('Are you Indigenous/Native American?',['Yes','No'])
if choice == 'Yes':
    st.write("It's your call.")
elif choice == 'No':
    st.write("Yeah, [it](https://www.cnn.com/2021/10/28/us/atlanta-braves-tomahawk-chop) [is](https://www.11alive.com/article/sports/mlb/atlanta-braves/atlanta-braves-tomahawk-chop-offensive/85-cb0c1d5f-123a-4f2e-9b74-b8798e451100).")

with st.bottom:
    st.caption("Powered by Woke 2.0")
