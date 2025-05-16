import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, f1_score
import itertools

# --- Data Generation Functions ---

def generate_simulated_detection_data(target_class_name, num_samples_per_class=35,
                                      true_positive_rate=0.88, # Adjusted for potentially harder real-world data
                                      true_negative_rate=0.92,
                                      latency_mean_ms=90, latency_std_ms=25):
    """
    Generates simulated data for a specific target class.
    """
    y_true = []
    y_scores = []
    latencies = [] # Latency will be generated per overall detection cycle

    # Target Class Samples (Positives)
    for _ in range(num_samples_per_class):
        y_true.append(1) # Is the target class
        if np.random.rand() < true_positive_rate:
            y_scores.append(np.random.uniform(0.65, 1.0))
        else: # Missed target (false negative)
            y_scores.append(np.random.uniform(0.1, 0.4))
        latencies.append(np.random.normal(latency_mean_ms, latency_std_ms))

    # Other Samples (Negatives for this specific target class)
    for _ in range(num_samples_per_class):
        y_true.append(0) # Is not the target class
        if np.random.rand() < true_negative_rate:
            y_scores.append(np.random.uniform(0.05, 0.35))
        else: # False positive (misclassified as target)
            y_scores.append(np.random.uniform(0.55, 0.85))
        latencies.append(np.random.normal(latency_mean_ms, latency_std_ms))
    
    latencies = [max(10, l) for l in latencies] # Ensure positive latency
    
    # Shuffle the data as it was generated in blocks
    indices = np.arange(len(y_true))
    np.random.shuffle(indices)
    y_true = np.array(y_true)[indices]
    y_scores = np.array(y_scores)[indices]
    # Latencies are associated with each detection attempt, so not shuffled relative to y_true/y_scores here
    # but we will collect all latencies for a general histogram

    return y_true, y_scores, np.array(latencies)

def generate_simulated_triangulation_data(num_samples=70, real_distances_range=(150, 1000), # Reduced samples, adjusted range
                                          dlt_error_percentage=0.05,
                                          sgbm_error_percentage_mean=0.10, # SGBM error mean (e.g., 10% overestimate)
                                          sgbm_error_percentage_std=0.22): # SGBM error standard deviation (larger variance)
    """
    Generates simulated data for triangulation performance.
    """
    real_distances = np.random.uniform(real_distances_range[0], real_distances_range[1], num_samples)
    real_distances = np.sort(real_distances)

    dlt_noise = np.random.uniform(-dlt_error_percentage, dlt_error_percentage, num_samples)
    dlt_calculated_distances = real_distances * (1 + dlt_noise)

    sgbm_percentage_errors = np.random.normal(sgbm_error_percentage_mean, sgbm_error_percentage_std, num_samples)
    sgbm_calculated_distances = real_distances * (1 + sgbm_percentage_errors)
    
    dlt_calculated_distances = np.maximum(dlt_calculated_distances, 10.0) 
    sgbm_calculated_distances = np.maximum(sgbm_calculated_distances, 10.0)

    return real_distances, dlt_calculated_distances, sgbm_calculated_distances

# --- Plotting Functions ---

def plot_pr_curve(y_true, y_scores, class_name, save_path_prefix="pr_curve"):
    """Plots and saves the Precision-Recall curve for a specific class."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    save_path = f"{save_path_prefix}_{class_name.lower().replace(' ', '_')}.png"

    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', color='b', alpha=0.7, label=f'AP ({class_name}) = {average_precision:0.2f}')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve for {class_name} Detection')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Precision-Recall curve for {class_name} saved to {save_path}")

def plot_detection_latency(all_latencies, save_path="overall_detection_latency.png"):
    """Plots and saves the histogram of overall detection latencies."""
    plt.figure(figsize=(8, 6))
    plt.hist(all_latencies, bins=20, color='lightcoral', edgecolor='black') # Changed color for distinction
    plt.xlabel('Detection Latency (ms)')
    plt.ylabel('Frequency')
    plt.title('Overall Histogram of Detection Latencies')
    mean_latency = np.mean(all_latencies)
    plt.axvline(mean_latency, color='darkred', linestyle='dashed', linewidth=1, label=f'Mean: {mean_latency:.2f} ms')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(save_path)
    plt.close()
    print(f"Overall detection latency histogram saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_name_positive, classes_display, normalize=False, title_suffix='Detection', cmap=plt.cm.Blues, save_path_prefix="confusion_matrix"):
    """
    This function prints and plots the confusion matrix for a specific class.
    `class_name_positive` is the name of the class we are focused on (e.g., 'Apple').
    `classes_display` should be like ['Not <Class>', '<Class>'].
    """
    cm = confusion_matrix(y_true, y_pred)
    save_path = f"{save_path_prefix}_{class_name_positive.lower().replace(' ', '_')}.png"
    
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'Confusion Matrix for {class_name_positive} {title_suffix}')
    plt.colorbar()
    tick_marks = np.arange(len(classes_display))
    plt.xticks(tick_marks, classes_display, rotation=45)
    plt.yticks(tick_marks, classes_display)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm # For text display, use normalized if available or original

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        display_val = f"{cm_normalized[i, j]:.2f}" if normalize else int(cm[i, j])
        plt.text(j, i, display_val,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix for {class_name_positive} saved to {save_path}")

def plot_metrics_vs_threshold(y_true, y_scores, class_name, save_path_prefix="metrics_vs_threshold"):
    """Plots Precision, Recall, and F1-score against varying confidence thresholds for a specific class."""
    thresholds = np.linspace(0.01, 1.0, 100)
    precisions = []
    recalls = []
    f1s = []
    save_path = f"{save_path_prefix}_{class_name.lower().replace(' ', '_')}.png"

    for thresh in thresholds:
        y_pred_thresh = (y_scores >= thresh).astype(int)
        tp = np.sum((y_pred_thresh == 1) & (y_true == 1))
        fp = np.sum((y_pred_thresh == 1) & (y_true == 0))
        fn = np.sum((y_pred_thresh == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    plt.figure(figsize=(10, 7))
    plt.plot(thresholds, precisions, label='Precision', color='blue', linestyle='--')
    plt.plot(thresholds, recalls, label='Recall', color='green', linestyle=':')
    plt.plot(thresholds, f1s, label='F1-Score', color='red')
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title(f'P, R, F1-Score vs. Threshold for {class_name}')
    plt.legend()
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.5)
    plt.savefig(save_path)
    plt.close()
    print(f"Metrics vs. Threshold plot for {class_name} saved to {save_path}")

def plot_distance_results(real_distances, dlt_distances, sgbm_distances, save_path_prefix="distance_analysis"):
    """
    Plots triangulation results:
    1. Calculated vs. Real distance scatter plot.
    2. Percentage error histograms for DLT and SGBM.
    """
    # 1. Calculated vs. Real Distance
    plt.figure(figsize=(10, 7))
    plt.scatter(real_distances, dlt_distances, alpha=0.7, edgecolors='k', s=50, label='DLT Calculated', c='blue')
    plt.scatter(real_distances, sgbm_distances, alpha=0.7, edgecolors='k', s=50, label='SGBM Calculated', c='red', marker='x')
    # Ideal line where calculated distance equals real distance
    min_val = min(np.min(real_distances), np.min(dlt_distances), np.min(sgbm_distances)) * 0.9
    max_val = max(np.max(real_distances), np.max(dlt_distances), np.max(sgbm_distances)) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal (Real Distance)')
    
    plt.xlabel('Real Distance (mm)')
    plt.ylabel('Calculated Distance (mm)')
    plt.title('Triangulation: Calculated vs. Real Distance')
    plt.legend()
    plt.grid(True)
    plt.xlim(left=min_val)
    plt.ylim(bottom=min_val)
    save_path_scatter = f"{save_path_prefix}_scatter.png"
    plt.savefig(save_path_scatter)
    plt.close()
    print(f"Distance scatter plot saved to {save_path_scatter}")

    # 2. Percentage Errors
    dlt_errors_percent = ((dlt_distances - real_distances) / real_distances) * 100
    sgbm_errors_percent = ((sgbm_distances - real_distances) / real_distances) * 100

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(dlt_errors_percent, bins=15, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('DLT Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title('DLT Distance Error Distribution')
    dlt_mean_err = np.mean(dlt_errors_percent)
    dlt_std_err = np.std(dlt_errors_percent)
    plt.axvline(dlt_mean_err, color='k', linestyle='dashed', linewidth=1, label=f'Mean: {dlt_mean_err:.2f}%')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)

    plt.subplot(1, 2, 2)
    plt.hist(sgbm_errors_percent, bins=15, color='red', alpha=0.7, edgecolor='black')
    plt.xlabel('SGBM Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title('SGBM Distance Error Distribution')
    sgbm_mean_err = np.mean(sgbm_errors_percent)
    sgbm_std_err = np.std(sgbm_errors_percent)
    plt.axvline(sgbm_mean_err, color='k', linestyle='dashed', linewidth=1, label=f'Mean: {sgbm_mean_err:.2f}%')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    plt.tight_layout()
    save_path_hist = f"{save_path_prefix}_error_hist.png"
    plt.savefig(save_path_hist)
    plt.close()
    print(f"Distance error histograms saved to {save_path_hist}")
    
    print(f"\nDLT Error Stats: Mean = {dlt_mean_err:.2f}%, Std = {dlt_std_err:.2f}%")
    print(f"SGBM Error Stats: Mean = {sgbm_mean_err:.2f}%, Std = {sgbm_std_err:.2f}%")

# --- Main Analysis Function ---

def run_experiment_analysis():
    """
    Runs the full analysis: generates data, plots figures, and prints summaries.
    """
    print("--- Running Experiment Analysis ---")
    
    target_object_classes = ["apple", "cup", "mouse"]
    all_simulated_latencies = []
    num_samples_per_class_for_detection = 35 # Approx 30-40 as requested

    print("\n[1] Simulating and Plotting Object Detection Performance (Per Class)...")
    for target_class in target_object_classes:
        print(f"\n  Analyzing class: {target_class.upper()}")
        y_true_cls, y_scores_cls, latencies_cls = generate_simulated_detection_data(
            target_class_name=target_class,
            num_samples_per_class=num_samples_per_class_for_detection,
            true_positive_rate=0.85, # Slightly lower TP rate for more realistic small dataset
            true_negative_rate=0.90  # Slightly lower TN rate
        )
        all_simulated_latencies.extend(latencies_cls)
        
        plot_pr_curve(y_true_cls, y_scores_cls, class_name=target_class, save_path_prefix="detection")
        
        chosen_threshold = 0.5 
        y_pred_cls = (y_scores_cls >= chosen_threshold).astype(int)
        cm_display_classes = [f'Not {target_class.capitalize()}', target_class.capitalize()]
        plot_confusion_matrix(y_true_cls, y_pred_cls, class_name_positive=target_class.capitalize(), 
                              classes_display=cm_display_classes, title_suffix=f'{target_class.capitalize()} Detection', 
                              save_path_prefix="detection_cm")
        plot_metrics_vs_threshold(y_true_cls, y_scores_cls, class_name=target_class.capitalize(), save_path_prefix="detection_metrics")

    # Plot overall latency from all detection simulations
    if all_simulated_latencies:
        plot_detection_latency(np.array(all_simulated_latencies), save_path="overall_detection_latency.png")

    # 2. Triangulation Performance
    print("\n[2] Simulating and Plotting Triangulation Performance...")
    # Using a total of 70 samples for triangulation, consistent with smaller dataset idea
    real_dists, dlt_dists, sgbm_dists = generate_simulated_triangulation_data(
        num_samples=70, # Adjusted to reflect smaller dataset scale
        real_distances_range=(150, 1000), 
        dlt_error_percentage=0.05,      
        sgbm_error_percentage_mean=0.10, # SGBM might have a systematic bias (e.g., 10% overestimate)
        sgbm_error_percentage_std=0.22  # SGBM has higher variability (std dev of 22%)
    )
    plot_distance_results(real_dists, dlt_dists, sgbm_dists, save_path_prefix="triangulation_analysis")

    print("\n--- Analysis Complete ---")
    print("Figures have been saved to the current directory.")
    print("Make sure you have matplotlib and scikit-learn installed:")
    print("  pip install matplotlib scikit-learn")
    print("\nNote: For a comprehensive paper, also include qualitative results, i.e., example images showing successful detections, false positives, and false negatives from your actual system for each class.")

if __name__ == '__main__':
    run_experiment_analysis() 