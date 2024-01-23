import os
import json
import matplotlib.pyplot as plt


def read_train_val_results(tests):
    all_results = []

    for t in tests:
        script_directory = os.path.dirname(os.path.realpath(__file__))
        test_file_name = os.path.join(
            script_directory, '..', 'results', t, 'results', 'tr_val_results.json')
        print(test_file_name)
        if os.path.exists(test_file_name):
            with open(test_file_name, 'r') as file:
                try:
                    data = json.load(file)
                    all_results.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {test_file_name}: {e}")
        else:
            print(f"Test results for {t} don't exist")
    return all_results


def create_line_plots(metrics, data, models_name, configuration, save_plot_prefix="plot"):
    script_directory = os.path.dirname(__file__)
    num_classes = len(data[0][0]["training_classes_metrics"])  # Assuming all runs have the same number of classes

    for metric in metrics:
        for class_idx in range(num_classes):
            fig, ax = plt.subplots(figsize=(10, 7))
            lines = []
            labels = []

            for j, model_data in enumerate(data):
                train_label = f"{models_name[j]} Train {metric[1]} Class {class_idx}"
                val_label = f"{models_name[j]} Validation {metric[1]} Class {class_idx}"

                train_values = [epoch[f'training_classes_metrics'][str(class_idx)][f'{metric[0]}']
                                for epoch in model_data]
                val_values = [epoch[f'validation_classes_metrics'][str(class_idx)][f'{metric[0]}']
                              for epoch in model_data]

                line_train, = ax.plot(range(1, len(train_values) + 1),
                                      train_values, marker='o', label=train_label)
                line_val, = ax.plot(range(1, len(val_values) + 1),
                                    val_values, marker='o', label=val_label, linestyle='dashed')

                lines.extend([line_train, line_val])
                labels.extend([train_label, val_label])

            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel(f'{metric[1]} Class {class_idx}', fontsize=14)
            ax.set_title(f'{metric[1]} Class {class_idx} Train and Validation Results', fontsize=16)
            ax.legend(lines, labels, loc='best')

            # Add a description under the title
            ax.text(0.5, -0.12, configuration, ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='black')

            # Save the plot to a file
            save_path = os.path.join(
                script_directory, f"{save_plot_prefix}_{metric[0]}_class_{class_idx}.png")
            plt.savefig(save_path)
            print(f"Plot saved as {save_path}")
            plt.close()  # Close the current plot before creating the next one


# ---CONFIGURATIONS---#
test_folders = [
    "mslanet"
]
metrics = [('accuracy', 'Accuracy'), ('sensitivity', 'Sensitivity'), ('auc', 'AUC'), ('specificity', 'Specificity')]
models_name = ["MSLANet v2"]
batch_size = 256
configuration = f"Segmentation=SAM, LR=1e-5, Batch Size={batch_size}"


assert len(test_folders) == len(
    models_name), "The number of tests and their name must be of equal length"

data = read_train_val_results(test_folders)
print(data)
create_line_plots(metrics, data, models_name, configuration)
