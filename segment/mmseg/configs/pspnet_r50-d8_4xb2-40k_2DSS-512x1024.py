_base_ = [
    'config/models/pspnet_r50-d8.py', 'datasets/2DSS.py',
    'default_runtime.py', 'schedules/schedule_40k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)