import streamlit as st
import pandas as pd
import joblib
from matplotlib import pyplot as plt


st.set_page_config(page_icon="🤖",page_title="Linear_reg")

st.title("Simple Linear Regression:")
st.write("Simple linear regression is a technique that uses a straight line to model the relationship between two continuous variables. It helps us understand how one variable (like year) might influence another (like income), and even predict values for unseen data points.")


df = pd.read_csv("canada_per_capita_income.csv")

# Display the subheader
st.subheader("Here we have an example dataset")

# Function to display the table
def tabel(data):
    st.table(data)


table_container = st.empty()

# Initialize session state to keep track of the table state
if 'show_more' not in st.session_state:
    st.session_state.show_more = False

# Display the table based on the state
if st.session_state.show_more:
    table_container.table(df)
else:
    table_container.table(df.head(5))

# Create two columns for the buttons and place them below the table
col1, col2 = st.columns(2)

with col1:
    if st.button(label="Show more"):
        st.session_state.show_more = True
        table_container.table(df)  # Update the table to show more rows

with col2:
    if st.button(label="Show less"):
        st.session_state.show_more = False
        table_container.table(df.head(5))  # Update the table to show fewer rows

model = joblib.load("model.jolib")
# Prepare the plot using Matplotlib
fig, ax = plt.subplots()  # Create a figure and axes
ax.scatter(df['year'], df['per capita income (US$)'], color='g', marker='*')
ax.set_ylabel('Per Capita Income (US$)')  # Set labels using ax object
ax.set_xlabel('Year')

# Display the plot in Streamlit
st.subheader("Scatter diagram of the dataset")
st.pyplot(fig)

fig, ax = plt.subplots()  # Create a figure and axes
ax.scatter(df['year'], df['per capita income (US$)'], color='b', marker='*')
ax.set_ylabel('Per Capita Income (US$)')  # Set labels using ax object
ax.set_xlabel('Year')

st.subheader("After model training")
st.code("""
reg = linear_model.LinearRegression()
model = reg.fit(df[['year']],df[['per capita income (US$)']])
""")

ax.plot(df['year'],model.predict(df[['year']]), color = 'r')

# Display the plot in Streamlit
st.pyplot(fig)

st.write("Intercept:",model.intercept_)
st.write("Coefficent:",model.coef_)

st.subheader("Predection")
inp = st.text_input(label="Enter year to predict per capita income (US$)")
pred = st.button(label="Predict")


if pred:
    if inp:
        inp = int(inp)
        st.write(model.predict([[inp]]))

    else:
        st.warning("Enter a valid number")


# inp = int(inp)
# st.write(model.predict([[inp]]))
