import pandas as pd

# Load the customer support tickets dataset from CSV
df = pd.read_csv('data/customer_support_tickets.csv')

# Print basic information: shape of the dataframe
print("Dataset Shape (Rows, Columns):", df.shape)
print()

# Print data types of each column
print("Column Data Types:")
print(df.dtypes)
print()

# Print count of missing values per column
print("Missing Values Count per Column:")
print(df.isnull().sum())
print()

# Print normalized value counts for Ticket Type (department/queue)
print("Ticket Type Distribution (Normalized):")
print(df['Ticket_Type'].value_counts(normalize=True))
print()

# Print normalized value counts for Ticket Priority
print("Ticket Priority Distribution (Normalized):")
print(df['Ticket_Priority'].value_counts(normalize=True))
print()

# Calculate average length of Body_Text in characters
avg_char_length = df['Body_Text'].str.len().mean()
print(f"Average Ticket Body Text Length (Characters): {avg_char_length:.2f}")

# Calculate average length of Body_Text in words
avg_word_length = df['Body_Text'].str.split().str.len().mean()
print(f"Average Ticket Body Text Length (Words): {avg_word_length:.2f}")
