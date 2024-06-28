import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from collections import defaultdict 


# Function to fetch data from Yahoo Finance
def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Function to preprocess the data
def preprocess_data(data):
    data['Date'] = data.index
    data = data.reset_index(drop=True)
    return data

# Function to train the linear regression model
def train_linear_model(data):
    X = data[['Open', 'High', 'Low', 'Volume']]  # Features
    y = data['Close']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test
# Function to train the Random Forest regression model
def train_random_forest_model(data):
    X = data[['Open', 'High', 'Low', 'Volume']]  # Features
    y = data['Close']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Function to make predictions
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Function to display the app homepage
def home_page():
    st.title("Stock Price Prediction ")
    
    # Display the logo
    st.subheader("Stock Logo")
    logo = mpimg.imread('companylogo.jpeg')
    st.image(logo, caption='Our Logo', width=200)
    
    st.subheader("Welcome to our Stock Price Prediction App!")
    st.write("Please select your username and enter the password to access the predictions.")

    # Initialize users dictionary in session state
    if 'users' not in st.session_state:
        st.session_state.users = {}
        st.session_state.admin_password = "12345"  # Set admin password

    # Admin section
    st.subheader("Admin Section")
    admin_password = st.text_input("Admin Password", type="password")
    if admin_password == st.session_state.admin_password:
        st.write("Admin Logged In")
        st.write("Current Sign-up Information:")
        for username, password in st.session_state.users.items():
            st.write(f"- Username: {username}, Password: {password}")
        st.subheader("Remove User")
        remove_username = st.text_input("Username to Remove")
        if st.button("Remove"):
            if remove_username in st.session_state.users:
                del st.session_state.users[remove_username]
                st.success(f"User '{remove_username}' removed successfully.")
            else:
                st.warning(f"User '{remove_username}' not found.")
    elif admin_password != "":
        st.warning("Incorrect admin password.")

    # Divider
    st.markdown("---")

    # Sign-up section
    st.subheader("Sign-up")
    sign_up_username = st.text_input("Sign-up Username")
    sign_up_password = st.text_input("Sign-up Password", type="password")
    if st.button("Sign Up"):
        if sign_up_username and sign_up_password:
            sign_up(sign_up_username, sign_up_password)
            # Clear input boxes after sign up
            sign_up_username = ""
            sign_up_password = ""
        else:
            st.warning("Please enter a username and password.")

    # Divider
    st.markdown("---")

    # Login section
    st.subheader("Login")
    login_username = st.text_input("Login Username")
    login_password = st.text_input("Login Password", type="password")
    if st.button("Login"):
        if login_username and login_password:
            login(login_username, login_password)
            # Clear input boxes after login
            login_username = ""
            login_password = ""
        else:
            st.warning("Please enter a username and password.")

# Function to sign up a new user
def sign_up(username, password):
    st.session_state.users[username] = password
    st.success("User signed up successfully.")

# Function to log in a user
def login(username, password):
    if username in st.session_state.users:
        if st.session_state.users[username] == password:
            st.session_state.is_logged_in = True
            st.success("Login successful.")
        else:
            st.error("Incorrect password.")
    else:
        st.error("Username not found.")


# Function to display the predict page
def predict_page():
    if "is_logged_in" not in st.session_state or not st.session_state.is_logged_in:
        st.error("You must login to access this page.")
        return

    st.title("Predict Stock Price")
    st.subheader("Enter the stock symbol and date range:")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
    start_date = st.date_input("Start Date:", pd.to_datetime('2022-01-01'))
    end_date = st.date_input("End Date:", pd.to_datetime('today'))

    # Fetching data
    data = get_stock_data(symbol, start_date, end_date)
    if data.empty:
        st.warning("No data found for the given symbol and date range. Please try again.")
        return

    # Preprocessing data
    data = preprocess_data(data)

    # Train the linear regression model
    linear_model, X_test, y_test = train_linear_model(data)
    linear_predictions = make_predictions(linear_model, X_test)

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, linear_predictions)
    st.write(f'Mean Squared Error: {mse}')

    # Display the logo
    st.subheader("Stock Logo")
    logo = mpimg.imread('logo.jpeg')
    st.image(logo, caption='Our Logo', width=200)

    # Display predicted price
    st.subheader("Predicted Price")
    st.write("Predicted closing price using Linear Regression model:")
    st.write(linear_predictions[-1])  # Show the latest predicted price

    # Provide advice for investors based on predicted price
    if len(linear_predictions) > 1:
        previous_price = linear_predictions[-2]  # Get the previous predicted price
        current_price = linear_predictions[-1]  # Get the latest predicted price
        price_change = current_price - previous_price
        if price_change > 0:
            st.write("The predicted price has increased since the last prediction.")
            st.write("Consider holding or buying the stock.")
        elif price_change < 0:
            st.write("The predicted price has decreased since the last prediction.")
            st.write("Consider selling or avoiding the stock.")
        else:
            st.write("The predicted price remains unchanged since the last prediction.")
            st.write("Consider monitoring the stock for further developments.")
    else:
        st.write("Not enough data for comparison. Please check back later.")

    # Bar graph of predicted prices
    st.subheader("Bar Graph of Predicted Prices")
    fig, ax = plt.subplots()
    ax.bar(range(len(linear_predictions)), linear_predictions, color='blue')
    ax.set_xlabel('Days')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Predicted Prices Over Days')
    st.pyplot(fig)

    # Line chart of predicted prices
    st.subheader("Line Chart of Predicted Prices")
    fig, ax = plt.subplots()
    ax.plot(range(len(linear_predictions)), linear_predictions, color='green')
    ax.set_xlabel('Days')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Predicted Prices Over Days')
    st.pyplot(fig)

    # Scatter plot of Actual vs Predicted price
    st.subheader("Actual vs Predicted Price")
    st.write("Scatter plot of Actual closing price vs Predicted closing price using Linear Regression model:")
    fig, ax = plt.subplots()
    ax.scatter(y_test, linear_predictions, color='blue')
    ax.plot(y_test, y_test, color='red', linewidth=2)
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Actual vs Predicted Price')
    st.pyplot(fig)

    # Line plot of actual prices
    st.subheader("Line Plot of Actual Prices")
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'], color='green')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('Line Plot of Actual Prices')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    # Advice for investors
    st.subheader("Advice for Investors")
    st.write("Based on the predictions and analysis, here are some recommendations for investors:")
    st.write("- *Short-term Investors*: Consider the predicted price trend over the short term (e.g., next few days) for potential trading opportunities.")
    st.write("- *Long-term Investors*: Focus on the overall trend and fundamentals of the company rather than short-term fluctuations.")
    st.write("- *Risk Management*: Always diversify your investment portfolio and use risk management strategies to mitigate potential losses.")
    st.write("- *Keep Informed*: Stay updated with the latest news and developments related to the company and the stock market.")

# Function to display the about page
def about_page():
    st.title("About Us: Stock Price System Company in Nairobi, Kenya CBD")
    
    st.subheader("Mission:")
    st.write("At Stock Price System Company, our mission is to empower investors in Nairobi's Central Business District (CBD) with reliable and actionable stock market insights. We strive to provide innovative solutions and comprehensive services that enable our clients to make informed investment decisions and achieve their financial goals.")
    
    st.subheader("Motto:")
    st.write("Empowering Investors, Driving Financial Success")
    
    st.subheader("Goals:")    
    st.write("1. Provide Accurate Market Analysis: We aim to deliver accurate and timely market analysis, leveraging advanced technologies and robust data analytics tools. By staying ahead of market trends and fluctuations, we empower investors to make informed decisions.")
    st.write("2. Offer Tailored Investment Solutions: Our goal is to offer personalized investment solutions tailored to the unique needs and preferences of each client. Whether they are new to investing or seasoned professionals, we provide guidance and support every step of the way.")
    st.write("3. Promote Financial Literacy: We are committed to promoting financial literacy and education among investors in Nairobi's CBD. Through workshops, seminars, and educational resources, we empower individuals to understand the intricacies of the stock market and make confident investment choices.")
    st.write("4. Ensure Transparency and Integrity: Transparency and integrity are at the core of our operations. We prioritize honesty, ethics, and accountability in all our dealings, fostering trust and confidence among our clients and stakeholders.")
    st.write("5. Drive Innovation in Financial Services: We strive to be at the forefront of innovation in financial services. By embracing new technologies and exploring cutting-edge solutions, we aim to enhance the efficiency, accessibility, and effectiveness of our services.")
    st.write("6. Support Economic Growth: As a local company based in Nairobi's CBD, we are dedicated to supporting economic growth and development in Kenya. By promoting investment opportunities and fostering entrepreneurship, we contribute to the prosperity of our community and nation.")
    st.write("7. Deliver Exceptional Customer Service: Customer satisfaction is paramount to us. We are committed to delivering exceptional customer service, ensuring that every client receives personalized attention, prompt assistance, and reliable support.")
    
     # Display team images
    st.subheader("Our Team")
    st.write("displays:")
    
    st.image("image 1.jpeg", caption="SHOW 1", width=150)
    st.image("image 2.jpeg", caption="SHOW 2", width=150)
    st.image("image 3.jpeg", caption="SHOW 3", width=150)
    st.image("image 4.jpeg", caption="SHOW 4", width=150)
 

    # Display client reviews
    st.subheader("Client Reviews")
    st.write("Here are some reviews from our satisfied clients:")
    
    # Client review cards
    reviews = [
        ("John Doe", "I found trading with Stock Price System Company to be very profitable and their insights were invaluable."),
        ("Jane Smith", "The team at Stock Price System Company provided excellent support and guidance throughout my investment journey."),
        ("David Johnson", "I highly recommend Stock Price System Company to anyone looking to invest in the stock market.")
    ]

    for name, review_text in reviews:
        display_review(name, review_text)

    # Allow clients to write their reviews
    st.subheader("Write Your Review")
    new_name = st.text_input("Your Name")
    new_review = st.text_area("Your Review", max_chars=300)
    if st.button("Submit Review"):
        if new_name.strip() and new_review.strip():
            display_review(new_name, new_review)
        else:
            st.warning("Please enter your name and review.")

def display_review(name, review_text):
    st.markdown(
        f"""
        <div style="background-color: #f5f5f5; border-radius: 10px; padding: 20px; margin: 10px; box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.2);">
            <p style="font-size: 18px; font-weight: bold; color: #333;">{name} 😊</p>
            <p style="font-size: 16px; color: #555;">"{review_text}"</p>
            <div style="display:flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                <div>
                    <button style="background-color: #82d173; color: white; border: none; border-radius: 5px; padding: 8px 12px; cursor: pointer;">😍 Like</button>
                    <button style="background-color: #e57373; color: white; border: none; border-radius: 5px; padding: 8px 12px; cursor: pointer;">😒 Dislike</button>
                </div>
                <div style="font-size: 14px; color: #888;">Share 🚀</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to display the contact page
def contact_page():
    
    # Display contact information
    st.write("If you have any questions or feedback, please contact us at:")
    st.write("Email: [info@stockpriceapp.com](mailto:info@stockpriceapp.com)")
    st.write("Twitter: [@stockpriceapp](https://twitter.com/stockpriceapp)")
    st.write("LinkedIn: [Stock Price App](https://www.linkedin.com/company/stockpriceapp)")
    st.write("Phone: 0748788876 , 0705339615")

    # Contact form
    st.subheader("Send Us a Message")
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Your Message", height=150)
    if st.button("Send"):
        if name.strip() and email.strip() and message.strip():
            # Add code to send the message (e.g., send an email)
            st.success("Your message has been sent!")
        else:
            st.warning("Please fill out all fields before sending.")

    # Interactive map
    st.subheader("Our Location")
    st.write("cbd Main Street, nairobi, kenya")

    # Social media links
    st.subheader("Connect With Us")
    st.write("Follow us on social media:")
    st.markdown("[![Twitter](https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white)](https://twitter.com/stockpriceapp)")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/company/stockpriceapp)")

    # Live chat (dummy button for demonstration)
    st.subheader("Chat With Us")
    st.write("Chat live with our support team:")
   

    # FAQ section (dummy content for demonstration)
    st.subheader("Frequently Asked Questions")
    st.write("Check out our FAQs for quick answers:")
    st.markdown("- How do I sign up?")
    st.markdown("- What are your payment methods?")
    st.markdown("- Can I cancel my subscription?")

    # Feedback or rating system (dummy buttons for demonstration)
    st.subheader("Feedback")
    st.write("Let us know how we're doing:")
    st.button("Leave Feedback")
    def main():
      st.title("Rate Our Website")

    st.write("We'd love to hear your feedback! Please rate your experience with our website.")

    # Rating scale
    rating = st.slider("Rate from 1 to 5", 1, 5)

    # Feedback form
    feedback = st.text_area("Leave your feedback (optional)")

    # Submit button
    if st.button("Submit"):
        # Add code to save the rating and feedback
        st.success("Thank you for your feedback!")

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Navigation App", page_icon=":smiley:", layout="wide")

    # Custom CSS styling for the sidebar and main background color
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            color: #212529;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0px 5px 10px rgba(0,0,0,0.1);
        }
        .sidebar h3 {
            color: #007bff;
            font-size: 1.2rem;
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
        .sidebar .sidebar-select {
            border-radius: 5px;
        }
        .sidebar .sidebar-toggle {
            background-color: #007bff;
            border-radius: 5px;
            color: white;
        }
        .sidebar .sidebar-toggle:hover {
            background-color: #0056b3;
        }
      
    </style>
    """,
    unsafe_allow_html=True
)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("", ["Home", "Predict", "About", "Contact"])

    if selected_page == "Home":
        home_page()
    elif selected_page == "Predict":
        predict_page()
    elif selected_page == "About":
        about_page()
    elif selected_page == "Contact":
        contact_page()

if __name__ == "__main__":
    main()
