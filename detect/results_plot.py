from ultralytics.utils.plotting import plot_results

plot_results(
    file='./runs/detect/train/results.csv', 
    dir='', 
    segment=False, 
    pose=False, 
    classify=False, 
    on_plot=None
)