import streamlit as st

icon_path = "eye.png"
how_to_use_path = "res/HowToUse.png"

PAGES = {
    "Home": 'home',
    "App": 'app',
    "AboutUs": 'aboutus'
}

# Set page configuration
st.set_page_config(page_title="ClassiVision - Realtime Train, Preview & Effortless Download",
                   page_icon=icon_path,
                   layout="wide",
                   initial_sidebar_state="auto")

# CSS for styling
st.markdown(
    """
    <style>
    <head>
    <link rel="icon" href="logo.png" type="image/x-icon">
    </head>
        body {
    background-image: url("bg.jpg");
    background-size: cover;
    background-position: center; 
    background-repeat: no-repeat; 
    background-attachment: fixed;
    color: #fff; 
    font-family: 'Arial', sans-serif;
}
        # h2{
        # margin-top:80px;
        # }
        header {
            padding-top: -40px;
            font-size:44px;
            color:#ffffff;
            text-align: center;
            padding: 20px;
            margin-top: -80px;
            margin-bottom:
        }
        .logo {
            display: inline-block;
            vertical-align: left;
            align-items:left;
            margin-right: 10px;
        }
        
        .center-box {
            display: inline-block;
            margin-bottom:150px;
            justify-content: center;
            align-items: center;
            height: 20vh; /* Adjust height to center the box */
        }
        .classification-box {
            display: inline-block;  
            text-align: center;
            width: 250px;
            margin-bottom:30px;
            margin-left:30px;
            height: 250px;
            text-decoration:none;
            padding-top: 20px;
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .classification-box:hover {
            transform: scale(1.05);
            text-decoration:none;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
        }
        .classification-box img {
            width: 100%;
            display: inline-block;  
            height: 80%;
            object-fit: contain;
            text-decoration:none;
            border-radius: 15px 15px 0 0;
        }
        .classification-title {
            font-weight: bold;
            font-size: 18px;
            display: inline-block;  
            text-decoration:none;
            color: #333;
            padding: 10px;
            text-align:center;
            align-items:center;
            justify-content:center;
        }
         footer {
            padding-top:100px;
            justify-content: center;
            text-align: center;
            margin-top: 50px;
            color: #666;
            font-family: 'Arial', sans-serif; /* Default font, replace with unique fonts */
        }
        
        footer p {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            font-family: 'Poppins', sans-serif; /* Bold and unique font for "Follow me on" */
            margin-bottom: 30px;
        }
        
        span{
            text-align:center;
        }
        footer a {
            margin: 0 10px;
            color: #666;
            text-decoration: none;
            font-size: 18px;
        }
        footer a:hover {
            color: #000;
        }
        .social-icons {
            display: flex;
            justify-content: center;
            margin-top: 30px; /* Add margin for spacing */
            gap: 20px; /* Space between icons */
        }
        .social-icons img {
            width: 40px; /* Increased size for better visibility */
            height: 40px;
            padding: 10px;
            background: #fff; /* White background behind icons */
            border-radius: 40%; /* Makes the white background circular */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); /* Adds a subtle shadow for a floating effect */
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .social-icons img:hover {
            transform: scale(1); /* Slight zoom on hover */
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2); /* Enhanced shadow on hover */
        }
        
        .title {
            color:#ffffff;
            margin-top: -50px;
            display: inline-block;
            font-size: 44px;
            font-weight: bold;
            vertical-align: middle;
            margin-bottom:30px;
        }
        
        p follow_bottom{
        margin-left:200px;
        }
        </style>
        """, unsafe_allow_html=True)

st.markdown(
    """
    <div class="header"> 
    <div class="title">ClassiVision: Train, Predict & Export</div>
    </div>
    
    """
, unsafe_allow_html=True, help=None)

st.divider()

st.sidebar.success("Select the Page")

# Center Classification Box with External SVG icon
st.markdown(
    """
    <h2 class="offer">What we Offer</h2>
    <div class="center-box">
        <a href='#', class="classification-box">
                <img src="https://cdn-icons-png.flaticon.com/512/12492/12492310.png" alt="Image Classification" />
            <div class="classification-title">Image Classification</div>
        </a>    
        <div class="center-box">
        <a href='#', class="classification-box">
            <img src="https://cdn-icons-png.flaticon.com/512/2975/2975842.png" alt="Image Segmentation" />
            <div class="classification-title">Image Segmentation</div>
        </a> </div>

        
    """,
    unsafe_allow_html=True,
)

st.header("How to use?")


st.image(image=how_to_use_path, caption="Credit: teachablemachines", width=1100 )

# About Developer Section
st.header("***About Developer***", divider=True)
st.markdown(
    "Hello! My name is **Arth Vala**, and I am currently pursuing an Integrated MCA with a specialization in **Artificial Intelligence** at Parul University. I am deeply passionate about **Computer Vision** and Artificial Intelligence. I thrive on working with cutting-edge technologies like **Deep Learning**, **Machine Learning**, and **Neural Networks** to innovate impactful solutions.",
    unsafe_allow_html=True,
)

# Footer with social media links
st.markdown("""

<div class="footer">
    <p class="follow_bottom"> Follow me on</p>
    <div class="social-icons">
        <a href="https://www.linkedin.com/in/arthvala" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
        </a>
        <a href="https://github.com/CrewArth" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="GitHub">
        </a>
        <a href="https://www.youtube.com/CricketGuruji15" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/1384/1384060.png" alt="YouTube">
        </a>
        <a href="https://www.instagram.com/arthvala.15" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" alt="Instagram">
        </a>
    </div>
</div>
""", unsafe_allow_html=True)



