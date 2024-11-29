import streamlit as st
from streamlit_lottie import st_lottie
import json
import time


# Function to load Lottie from local JSON
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load Lottie Animation from local file
lottie_contact = load_lottiefile("handshake.json")  # Save handshake.json in the project folder

# Page Configuration
st.set_page_config(page_title="Contact Me", page_icon="📞", layout="centered")

with st.spinner('Page is Loading...'):
    time.sleep(1)


# Page Heading
st.markdown(
    """
    <h1 style="text-align: center; color: white;">Contact Me</h1>
    <p style="text-align: center; color: white; font-size: 18px;">
        Feel free to reach out! Whether you have a question, a project, or just want to say hello. 
    </p>
    """,
    unsafe_allow_html=True,
)

# Main Content Layout
with st.container():
    # Two Columns for Animation and Contact Info
    col1, col2 = st.columns([1, 1])

    with col1:
        # Display Animation
        st_lottie(lottie_contact, height=250, key="contact")

    with col2:
        # Contact Info
        st.markdown(
            """
            <div style="background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);">
                <h3 style="text-align: center; color: #333;">Reach Out On</h3>
                <ul style="list-style-type: none; padding: 0; text-align: center;">
                    <li><img src="https://cdn-icons-png.flaticon.com/512/145/145807.png" width="25" style="vertical-align: middle;"> <a href="https://www.linkedin.com/in/arthvala" target="_blank" style="text-decoration: none; color: black;">LinkedIn</a></li>
                    <li><img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="25" style="vertical-align: middle;"> <a href="mailto:arthvala@gmail.com/" style="text-decoration: none; color: black;">Email</a></li>
                    <li><img src="https://cdn-icons-png.flaticon.com/512/733/733558.png" width="25" style="vertical-align: middle;"> <a href="https://www.instagram.com/arthvala.15/" target="_blank" style="text-decoration: none; color: black;">Instagram</a></li>
                    <li><img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="25" style="vertical-align: middle;"> <a href="https://github.com/CrewArth/" target="_blank" style="text-decoration: none; color: black;">GitHub</a></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Contact Form
st.markdown("---")
st.markdown(
    """
    <h2 style="text-align: center; color: white;">Get in Touch</h2>
    <p style="text-align: center; color: white;">
        Leave your email and message below, and I'll get back to you soon!
    </p>
    """,
    unsafe_allow_html=True,
)

# Contact Form Fields
with st.form(key="contact_form"):
    email = st.text_input("Your Email", placeholder="Enter your email address")
    message = st.text_area("Your Message", placeholder="Write your message here...")
    submitted = st.form_submit_button("Submit")

    # Form Submission Response
    if submitted:
        if email and message:
            st.success("Thank you for reaching out! I'll get back to you shortly.")
        else:
            st.error("Please fill out both fields before submitting.")

if st.button("Back to Homepage"):
    st.switch_page("Home.py")

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; color: white; font-size: 14px;">
        Made with ❤️ by Arth Vala
    </div>
    """,
    unsafe_allow_html=True,
)
