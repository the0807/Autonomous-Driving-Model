_base_ = [
    'models/deeplabv3plus_r50-d8.py',
    'datasets/seg2DSS.py', 'default_runtime.py',
    'schedules/schedule_40k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)