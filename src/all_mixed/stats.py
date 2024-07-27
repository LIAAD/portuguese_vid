import os
import json
import plotly.express as px
import plotly.graph_objects as go


def print_to_file(results_df, test_accuracy, output_path, TRAIN_CONDITIONS, type_of_embeddings):

    if results_df is not None:
        results_df.to_json(os.path.join(output_path, "results.json"),
                           orient="records", indent=4, force_ascii=False)
    if test_accuracy is not None:
        TRAIN_CONDITIONS["test_accuracy"] = test_accuracy

        TRAIN_CONDITIONS["type_of_embeddings"] = type_of_embeddings

        with open(os.path.join(output_path, "test_accuracy.txt"), "w") as f:
            json.dump(TRAIN_CONDITIONS, f, indent=4, ensure_ascii=False)


def loss_chart(results_df, output_path):

    results_df = results_df.copy()

    results_df['epoch'] += 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=results_df['epoch'], y=results_df['train_loss'],
                  name='Train Loss', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=results_df['epoch'], y=results_df['validation_loss'],
                  name='Validation Loss', line=dict(color='red', width=3)))

    # Customize layout
    fig.update_layout(xaxis_title='Epoch', yaxis_title='Loss')

    # Update x-axis range to start from 1
    fig.update_xaxes(range=[1, max(results_df['epoch'])])

    fig.write_image(os.path.join(output_path, "loss_chart.pdf"))


def accuracy_chart(results_df, output_path):
    fig = px.line(results_df, x='epoch', y=[
                  'validation_accuracy'], title='Validation Accuracy Over Epochs')

    fig.update_traces(line=dict(color='blue'))

    # Customize layout
    fig.update_layout(xaxis_title='Epoch', yaxis_title='Accuracy')

    # Update x-axis range to start from 1
    fig.update_xaxes(range=[1, max(results_df['epoch'])])

    fig.write_image(os.path.join(output_path, "accuracy_chart.pdf"))


def plot_chart(results_df, output_path):
    loss_chart(results_df, output_path)
    accuracy_chart(results_df, output_path)
