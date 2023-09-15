import pandas as pd
import statsmodels.formula.api as smf
import requests
from io import StringIO
import graphing
import plotly.offline as pyo  # Import plotly's offline module
import plotly.io as pio
import joblib

# Download the graphing.py file
graphing_url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py"
response = requests.get(graphing_url)
if response.status_code == 200:
    with open("graphing.py", "w") as file:
        file.write(response.text)
else:
    print(f"Failed to download {graphing_url}")

# Download the data file and load it into a DataFrame
data_url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/doggy-boot-harness.csv"
response = requests.get(data_url)
if response.status_code == 200:
    data = pd.read_csv(StringIO(response.text))
else:
    print(f"Failed to download {data_url}")


# Make a dictionary of data for boot sizes and harness size in cm
data = {
    'boot_size': [39, 38, 37, 39, 38, 35, 37, 36, 35, 40,
                  40, 36, 38, 39, 42, 42, 36, 36, 35, 41,
                  42, 38, 37, 35, 40, 36, 35, 39, 41, 37,
                  35, 41, 39, 41, 42, 42, 36, 37, 37, 39,
                  42, 35, 36, 41, 41, 41, 39, 39, 35, 39
                 ],
    'harness_size': [58, 58, 52, 58, 57, 52, 55, 53, 49, 54,
                     59, 56, 53, 58, 57, 58, 56, 51, 50, 59,
                     59, 59, 55, 50, 55, 52, 53, 54, 61, 56,
                     55, 60, 57, 56, 61, 58, 53, 57, 57, 55,
                     60, 51, 52, 56, 55, 57, 58, 57, 51, 59
                    ]
}

# Convert it into a DataFrame using pandas
dataset = pd.DataFrame(data)

# Define the formula for the linear regression model
formula = "boot_size ~ harness_size"

# Create the model
model = smf.ols(formula=formula, data=dataset)

# Fit the model
fitted_model = model.fit()

# Print model parameters
print("The following model parameters have been found:")
print(f"Line slope: {fitted_model.params[1]}")
print(f"Line Intercept: {fitted_model.params[0]}")

# Show a graph of the result
scatter_plot = graphing.scatter_2D(dataset, label_x="harness_size", label_y="boot_size",
                     trendline=lambda x: fitted_model.params[1] * x + fitted_model.params[0])

# Save the plot as an image file (e.g., PNG)
pio.write_image(scatter_plot, './scatterPlot.png')

# Define a harness size for prediction
harness_size_to_predict = {'harness_size': [58]}

# Use the model to predict boot size
approximate_boot_size = fitted_model.predict(harness_size_to_predict)

# Print the result
print("Estimated approximate_boot_size:")
print(approximate_boot_size[0])


# Let's begin by opening the dataset from file.
# import pandas
# !pip install statsmodels
# !wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py
# !wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/doggy-boot-harness.csv

# Load a file containing dog's boot and harness sizes
data = pd.read_csv('doggy-boot-harness.csv')

# Print the first few rows
data.head()

# Create and train a model
# As we've done before, we'll create a simple Linear Regression model and train it on our dataset.
# import statsmodels.formula.api as smf

# Fit a simple model that finds a linear relationship
# between boot size and harness size, which we can use later
# to predict a dog's boot size, given their harness size
model = smf.ols(formula = "boot_size ~ harness_size", data = data).fit()

print("Model trained!")

# Save and load a model
# Our model is ready to use, but we don't need it yet. Let's save it to disk.
#import joblib      # install it first

model_filename = './avalanche_dog_boot_model.pkl'
joblib.dump(model, model_filename)

print("Model saved!")

# Loading our model is just as easy:
model_loaded = joblib.load(model_filename)

print("We have loaded a model with the following parameters:")
print(model_loaded.params)



# Put it together
# Let's write a function that loads and uses our model
def load_model_and_predict(harness_size):
    '''
    This function loads a pretrained model. It uses the model
    with the customer's dog's harness size to predict the size of
    boots that will fit that dog.

    harness_size: The dog harness size, in cm 
    '''

    # Load the model from file and print basic information about it
    loaded_model = joblib.load(model_filename)

    print("We've loaded a model with the following parameters:")
    print(loaded_model.params)

    # Prepare data for the model
    inputs = {"harness_size":[harness_size]} 

    # Use the model to make a prediction
    predicted_boot_size = loaded_model.predict(inputs)[0]
    
    return predicted_boot_size

# Practice using our model
predicted_boot_size = load_model_and_predict(45)

print("Predicted dog boot size:", predicted_boot_size)


# Real world use
# We've done it; we can predict an avalanche dog's boot size based on the size of their harness. Our last step is to use this to warn people if they might be buying the wrong sized doggy boots.
# As an example, we'll make a function that accepts the harness size, the size of the boots selected, and returns a message for the customer. We would integrate this function into our online store.

def check_size_of_boots(selected_harness_size, selected_boot_size):
    '''
    Calculates whether the customer has chosen a pair of doggy boots that 
    are a sensible size. This works by estimating the dog's actual boot 
    size from their harness size.

    This returns a message for the customer that should be shown before
    they complete their payment 

    selected_harness_size: The size of the harness the customer wants to buy
    selected_boot_size: The size of the doggy boots the customer wants to buy
    '''

    # Estimate the customer's dog's boot size
    estimated_boot_size = load_model_and_predict(selected_harness_size)

    # Round to the nearest whole number because we don't sell partial sizes
    estimated_boot_size = int(round(estimated_boot_size))

    # Check if the boot size selected is appropriate
    if selected_boot_size == estimated_boot_size:
        # The selected boots are probably OK
        return f"Great choice! We think these boots will fit your avalanche dog well."

    if selected_boot_size < estimated_boot_size:
        # Selected boots might be too small 
        return "The boots you have selected might be TOO SMALL for a dog as "\
               f"big as yours. We recommend a doggy boots size of {estimated_boot_size}."

    if selected_boot_size > estimated_boot_size:
        # Selected boots might be too big 
        return "The boots you have selected might be TOO BIG for a dog as "\
               f"small as yours. We recommend a doggy boots size of {estimated_boot_size}."
    

# Practice using our new warning system
check_size_of_boots(selected_harness_size=55, selected_boot_size=39)


