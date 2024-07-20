import json
import matplotlib.pyplot as plt

file_path = 'output/metrics.json'

with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]

iterations = [entry['iteration'] for entry in data]
total_losses = [entry['total_loss'] for entry in data]

plt.figure(figsize=(10, 6))
plt.plot(iterations, total_losses, marker='o', linestyle='-', color='b')
plt.xlabel('Iteration')
plt.ylabel('Total Loss')
plt.title('Total Loss')

plt.savefig('output/result_plot.png')