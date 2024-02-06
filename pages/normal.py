import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Function to generate data and plot histogram with normal distribution curve
def generate_plot(mean, std_dev):
    np.random.seed(42)  # Set seed for reproducibility
    data = np.random.normal(loc=mean, scale=std_dev, size=1000)

    # Create histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

    # Create normal distribution curve
    #xmin, xmax = plt.xlim()
    xmin, xmax = -200,200
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std_dev)
    plt.plot(x, p, 'k', linewidth=2)

    # Add labels and title
    plt.title(f'Histogram with Normal Distribution ($\mu$={mean}, $\sigma$={std_dev})')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Display the plot
    st.pyplot(plt)

# Streamlit app

st.title('Histogram with Normal Distribution Curve')

# Sidebar content
st.sidebar.header("About")
st.sidebar.info(
    "This app lets you vary the population mean and population standard deviation to  explore the characteristics of a normal distribution."
    "\n"
    "\nA project for learning statistics by Foo Choo Yen"
)
# Slider for population mean
mean = st.slider('Select Population Mean', -50.0, 50.0, 0.0, step=0.1)

# Slider for population standard deviation
std_dev = st.slider('Select Population Standard Deviation', 0.0, 50.0, 30.0, step=0.1)

# Generate and display the plot
generate_plot(mean, std_dev)

