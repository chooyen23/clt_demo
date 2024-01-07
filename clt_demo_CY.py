import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('Illustrating the Central Limit Theorem with Streamlit') 
st.subheader('An App by Foo Choo Yen') 
st.write('This app lets you visualize different distributions and explore their sample mean distributions.') 
st.write('Select the Base Distribution and followed by indicating the sample size.')
st.write('Play around with the sample size to see the changes in the distribution of the Sample Means.') 
# Dropdown for distribution selection
distribution = st.selectbox(
    'Select Base Distribution',
    ('Uniform', 'Right Skewed', 'Triangular', 'Normal')
)

# Parameters for generating distributions
num_values = 10000
if distribution == 'Uniform':
    base_dist = np.random.uniform(0, 100, num_values)
elif distribution == 'Right Skewed':
    mu, sigma = 0, 0.5  # Mean and standard deviation for lognormal
    base_dist = np.random.lognormal(mu, sigma, num_values)
elif distribution == 'Triangular':
    base_dist = np.random.triangular(left=0, mode=50, right=100, size=num_values)
elif distribution == 'Normal':
    mu, sigma = 50, 15  # Mean and standard deviation for normal
    base_dist = np.random.normal(mu, sigma, num_values)

# Plotting the selected distribution
fig, ax = plt.subplots()
ax.hist(base_dist, bins=30, color='skyblue', edgecolor='black')
ax.set_title(f'{distribution} Distribution')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# User input for number of samples
n_samples = st.number_input('Number of samples drawn (max:1000)', min_value=1, max_value=1000, value=30, step=1)

list_of_means = []
for i in range(n_samples):
    sample_mean = np.random.choice(base_dist, 100, replace=True).mean()
    list_of_means.append(sample_mean)

# Plotting the histogram of sample means
fig, ax = plt.subplots()
ax.hist(list_of_means, bins=20, color='lightgreen', edgecolor='black')
ax.set_title('Distribution of Sample Means')
ax.set_xlabel('Sample Mean')
ax.set_ylabel('Frequency')
st.pyplot(fig)

