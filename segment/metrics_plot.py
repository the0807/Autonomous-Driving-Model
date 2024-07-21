import json
import matplotlib.pyplot as plt

file_path = 'output/metrics.json'

with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]

iterations = [entry['iteration'] for entry in data]
total_losses = [entry['total_loss'] for entry in data]
accuracies = [entry.get('mask_rcnn/accuracy', 0) for entry in data]
false_negatives = [entry.get('mask_rcnn/false_negative', 0) for entry in data]
false_positives = [entry.get('mask_rcnn/false_positive', 0) for entry in data]

plt.figure(figsize=(18, 10))

# Total Loss plot
plt.subplot(2, 2, 1)
plt.plot(iterations, total_losses, marker='o', linestyle='-', color='b')
plt.xlabel('Iteration')
plt.ylabel('Total Loss')
plt.title('Iterations/Total Loss')

# Accuracy plot
plt.subplot(2, 2, 2)
plt.plot(iterations, accuracies, marker='o', linestyle='-', color='g')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Iterations/Accuracy')

# False Negative plot
plt.subplot(2, 2, 3)
plt.plot(iterations, false_negatives, marker='o', linestyle='-', color='r')
plt.xlabel('Iteration')
plt.ylabel('False Negative')
plt.title('Iterations/False Negative')

# False Positive plot
plt.subplot(2, 2, 4)
plt.plot(iterations, false_positives, marker='o', linestyle='-', color='m')
plt.xlabel('Iteration')
plt.ylabel('False Positive')
plt.title('Iterations/False Positive')

plt.tight_layout()
plt.savefig('output/metrics.png')
plt.show()
