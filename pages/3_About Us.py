import streamlit as st
import time


# Set page configuration
st.set_page_config(page_title="About Us", page_icon="👤", layout="wide")


with st.spinner('Page is Loading...'):
    time.sleep(1)



# Add custom CSS
st.markdown(
    """
    <style>
        .container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 50px;
            margin-bottom: 50px;
        }
        .content {
            flex: 1;
            margin-right: 50px;
        }
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f5f5f5;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            color: #555;
        }
        hr {
            border: none;
            border-top: 2px solid #ddd;
            margin: 20px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)





# Add About Us content
st.markdown("<h1 style='text-align: center;'>About Us</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Use columns for layout
col1, col2 = st.columns([2, 1])  # Adjust width ratio for content and image

# Content on the left
with col1:
    st.subheader("**About Arth Vala**")
    st.write(
        """
        My name is Arth Vala, and I am a final-year student at Parul University,
         pursuing an Integrated MCA with a specialization in Artificial Intelligence.
          Set to graduate in 2025, I am deeply passionate about advancing technologies 
          in Artificial Intelligence, Computer Vision, Deep Learning, and Machine Learning.
           My academic journey has been driven by a strong interest in exploring the potential 
           of AI and its transformative impact on real-world applications.

        I'm currently focused on building AI-powered applications, 
        contributing to open-source projects, and mentoring aspiring developers. 
        Feel free to connect with me on my journey!
        """
    )

# Image on the right
with col2:
    st.image(
        "AboutUsImage.png",  # Replace with your image path
        caption="Mr. Arth Vala",
        use_container_width =True,
    )

if st.button("Back to Homepage"):
    st.switch_page("Home.py")

# Footer
st.markdown(
    """
    <div class="footer">
        © 2024 Arth Vala. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)
