# auth.py
import bcrypt
import streamlit as st

def setup_authentication():
    """Handle user authentication via a simple login form."""
    if st.session_state.get("authentication_status"):
        return True

    login_container = st.empty()
    with login_container.container():
        st.title("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        login_button = st.button("Login", key="login_button")

        valid_credentials = {
            'fsahin': {
                'password': bcrypt.hashpw('potatoes'.encode('utf-8'), bcrypt.gensalt()),
                'name': 'Ferat Sahin'
            },
            'ksubramanian': {
                'password': bcrypt.hashpw('25911'.encode('utf-8'), bcrypt.gensalt()),
                'name': 'Karthik Subramanian'
            },

            'oadamides': {
                'password': bcrypt.hashpw('FocusRs!'.encode('utf-8'), bcrypt.gensalt()),
                'name': 'Eddie Adamides'
            }
        }

        if login_button:
            if username in valid_credentials:
                hashed_password = valid_credentials[username]['password']
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                    st.session_state.authentication_status = True
                    st.session_state.username = username
                    st.session_state.name = valid_credentials[username]['name']
                    st.success(f"Welcome {valid_credentials[username]['name']}!")
                    login_container.empty()
                    return True
                else:
                    st.error("Invalid password")
            else:
                st.error("Invalid username")
    return False
