import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(data, title, x_label, save_path, show=False):
    fig = px.line(data, x=x_label, y="f1", title=title)
    fig.add_scatter(x=data[x_label], y=data["f1"], mode="lines", name="f1")
    fig.add_scatter(x=data[x_label], y=data["precision"], mode="lines", name="precision")
    fig.add_scatter(x=data[x_label], y=data["recall"], mode="lines", name="recall")
    # make y axis range from 0 to 1
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="score",
        legend_title="score type",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="Black"
        )
    )
    if save_path:
        fig.write_image(save_path)
    if show:
        fig.show()

# Get data from csv file
file_path = "tm_20newsgroups_average_smwond.csv"
data = pd.read_csv(file_path)

# Plot data
plot_data(data, "Average P/R/F1 for missing word prediction with<br> positional indexing on test data optimized no data drop",
          "epoch", save_path="tm_20ng_smwond_train.png", show=True)
