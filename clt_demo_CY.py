import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.title('Illustrating the Central Limit Theorem') 

st.write('Select the Base Distribution and followed by indicating the sample size.')
st.write('Play around with the sample size to see the changes in the distribution of the Sample Means.') 

# Sidebar content
st.sidebar.header("About")
st.sidebar.info(
    "This app lets you visualize different distributions and explore their sample mean distributions.\n"
    "A project for learning statistics by Foo Choo Yen"
)
# Dropdown for distribution selection

@st.cache_resource

def generate_base_distribution(distribution, num_values):
    if distribution == 'Uniform':
        return np.random.uniform(0, 100, num_values)
    elif distribution == 'Right Skewed':
        mu, sigma = 0, 0.5  # Mean and standard deviation for lognormal
        return np.random.lognormal(mu, sigma, num_values)
    elif distribution == 'Left Skewed':
        alpha, beta = 2, 20  # These can be adjusted
        return -np.random.beta(alpha, beta, num_values)
    elif distribution == 'Triangular':
        return np.random.triangular(left=0, mode=50, right=100, size=num_values)
    elif distribution == 'Normal':
        mu, sigma = 50, 15  # Mean and standard deviation for normal
        return np.random.normal(mu, sigma, num_values)
    
    elif distribution == 'Normal':
        mu, sigma = 50, 15  # Mean and standard deviation for normal
        return np.random.normal(mu, sigma, num_values)

    elif distribution == 'Exponential':
        a =2
        return np.random.exponential( scale=a,size=num_values)

def plot_distribution(data, title, bins=30, color='skyblue'):
    fig, ax = plt.subplots()

    # Plot histogram
    ax.hist(data, bins=bins, color=color, edgecolor='black')
    title_str = f"{distribution}"
    ax.set_title(title_str)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    return fig

def plot_sampling_distribution(data, title, bins=30, color='skyblue'):
    fig, ax = plt.subplots()
    ax.set_title(title)

    # Plot histogram
    ax.hist(data, bins=bins, color=color, edgecolor='black',density=False)

    # Calculate mean and standard deviation
    mean, std = np.mean(data), np.std(data)

    # Plot the normal distribution
    #xmin, xmax = ax.get_xlim()
    #x = np.linspace(xmin, xmax, 100)
    #p = norm.pdf(x, mean, std)
    #ax.plot(x, p, 'k', linewidth=2)

   # title_str = f"{title}\nMean: {mean:.2f}, Std Error: {std:.2f}"
    #ax.set_title(title_str)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    return fig

st.title('Distribution Visualization')

# Dropdown for distribution selection
distribution = st.selectbox('Select Base Distribution', ['Uniform', 'Right Skewed', 'Left Skewed','Triangular', 'Normal','Exponential'],
                            key='unique_distribution_select')

# Parameters for generating distributions
num_values = 1000
base_dist = generate_base_distribution(distribution, num_values)

# Plotting the selected distribution
fig = plot_distribution(base_dist, f'{distribution} Distribution')
st.pyplot(fig)

# User input for number of samples
n_samples = st.slider('Sample size (n)', min_value=1, max_value=1000, value=30, step=1)

# Generating sample means
list_of_means = []
for i in range(500):
    sample_mean = np.random.choice(base_dist, n_samples, replace=True).mean()
    list_of_means.append(sample_mean)
#list_of_means = [np.random.choice(base_dist,replace=True).mean() for _ in range(n_samples)]

mu_x_bar=np.mean(list_of_means)
sigma_x_bar = np.std(list_of_means)
# Plotting the histogram of sample means
title = f'Distribution of sample mean ($\mu_{{\\bar{{x}}}}$  = {mu_x_bar:0.2f}, $\sigma_{{\\bar{{x}}}}$ ={sigma_x_bar:0.2f} )'
mean_fig = plot_sampling_distribution(list_of_means, title, color='lightgreen')
st.pyplot(mean_fig,)
