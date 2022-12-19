import pandas as pd


# Create a sample dataframe
df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})

# Define the function you want to apply
def myfunc(x):
  return x * 2

# Apply the function to the desired column
df['col1'] = df['col1'].apply(myfunc)

# The resulting dataframe will have the values in 'col1' transformed by your function
print(df)
