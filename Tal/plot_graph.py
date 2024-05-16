import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../outputs/acs/all_css.csv")

plt.hist(df["css_2"], color='skyblue', edgecolor='black')
# Add labels and title
plt.xlabel('iscore')
plt.ylabel('Frequency')
plt.title("Histogram of iscore")

# Show the plot
plt.show()
