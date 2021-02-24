# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 2500

# How often to save a model checkpoint
SAVE_INTERVAL = 2500

def fake_parse():
    FLAGS = {
        'trace': "./",
        'num_iterations': 300000,
        'pretrained_model': '',
        'mode': 'flow',
        'train_test': 'train',
        'retrain': True,
        'data_dir': 'kitti_data',
        'train_file':'./filenames/kitti_train_files_png_4frames.txt',
        'gt_2012_dir': '',
        'gt_2015_dir': '',
        'batch_size': 4,
        'learning_rate': 0.0001,
        'num_gpus': 1,
        "img_height": 256,
        "img_width": 832,
        "depth_smooth_weight": 10.0,
        "ssim_weight": 0.85,
        "flow_smooth_weight": 10.0,
        "flow_consist_weight": 0.01,
        "flow_diff_threshold": 4.0,
        'eval_pose': '',
        'num_scales': 4,
    }

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    options = Struct(**FLAGS)
    return options