from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Plotting scatter plots to check the relationships between petal-lenght 
    and the species of iris flower.
    """
    # Creates a figure (fig) and an axis (ax) for plotting. 
    # dpi = 144 Increases the resolution to display better.
    fig, ax = plt.subplots(dpi=144)
    # plotting a scatter plot, the label was used to produce a good legend
    ax.scatter(x=df.petal_length, y=df.petal_width, color='purple')
    # Dynamic axis labels
    ax.set_xlabel(df.columns[2])
    ax.set_ylabel(df.columns[3])
    ax.set_title("Petal Length vs Petal Width")
    # Save and show the plot
    plt.savefig('relational_plot.png')
    plt.show()
    return


def plot_categorical_plot(df):
    """
    Plots a pie chart showing the distribution of different Iris species.
    """
    # Use group-by to count each species
    species_counts = df.groupby('species').size()
    # Create figure and axis
    fig, ax = plt.subplots(dpi=144)
    # Plot Pie Chart using `ax`
    species_counts.plot(
        ax=ax, kind='pie', autopct='%1.1f%%', startangle=140,
        colors=['lightblue', 'lightgreen', 'lightcoral']
    )
    # Set title and remove y-label for clarity
    ax.set_title("Distribution of Iris Species")
    ax.set_ylabel('')
    # Ensure the pie chart is a perfect circle
    ax.axis('equal')
    # Save the plot before displaying
    plt.savefig('categorical_plot.png')
    # Show the plot
    plt.show()
    return


def plot_statistical_plot(df):
    """
    Plots a box plot showing petal length distribution for each species.
    """
    # Create figure and axis
    fig, ax = plt.subplots(dpi=144)
    # Plot the box plot on the `ax` object
    sns.boxplot(
        ax=ax, data=df, x="species", y="petal_length", hue="species",
        palette="Set2", legend=False
    )
    # Set title and axis labels
    ax.set_title("Petal Length Distribution by Species")
    ax.set_xlabel("Species")
    ax.set_ylabel("Petal Length")
    # Save the plot before displaying
    plt.savefig('statistical_plot.png')
    # Show the plot
    plt.show()
    return


def statistical_analysis(df, col: str):
    """
    Computes statistical properties (mean, standard deviation, skewness, and kurtosis)
    for a given numerical column in the DataFrame.
    """
    # Compute statistics
    mean = df[col].mean()  # Mean (average)
    stddev = df[col].std() # Standard Deviation (spread of values)
    skew = ss.skew(df[col], nan_policy='omit')  # Skewness (asymmetry)
    excess_kurtosis = ss.kurtosis(df[col], nan_policy='omit')  # Excess Kurtosis
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Cleans the dataset by removing duplicates, handling missing values,
    and printing key insights such as summary statistics, 
    first few rows, and correlation matrix.
    """
    # Drop duplicate rows if any
    # df = df.drop_duplicate()
    # Drop rows with missing values if any
    df = df.dropna()
    # make use of quick features such as 'describe', 'head/tail' and 'corr'.
    # Display basic summary statistics
    print("Basic summary of the data:\n", df.describe())
    # Displays the first five rows of the data by default
    print("\nHead(the first five rows):\n", df.head())
    # Computes correlation matrix for numerical columns
    print("\nThe correlations of numerical data:\n", df.corr(numeric_only=True))
    # Returns the preprocessed data
    return df


def writing(moments, col):
    """
    Prints statistical moments and interprets skewness 
    and kurtosis of a dataset.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    # Delete the following options as appropriate for your data.
    # Not skewed and mesokurtic can be defined with asymmetries <-2 or >2.
    # print('The data was right/left/not skewed and platy/meso/leptokurtic.')
    # Interpret skewness based on asymmetry definition <-2 or >2
    if moments[2] < -2:
        skewness_type = "left-skewed"
    elif moments[2] > 2:
        skewness_type = "right-skewed"
    else:
        skewness_type = "not skewed"
    # Interpret kurtosis based on defined criteria
    if moments[3] < -1:
        kurtosis_type = "platykurtic"
    elif moments[3] > 1:
        kurtosis_type = "leptokurtic"
    else:
        kurtosis_type = "mesokurtic"

    print(f'The data is {skewness_type} and {kurtosis_type}.')
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'petal_length'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()