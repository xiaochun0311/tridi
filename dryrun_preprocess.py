import sys
from types import SimpleNamespace
sys.path.insert(0, "/home/yanghc/tridi")
from tridi.preprocessing.common import init_preprocessing

args = SimpleNamespace(
    config=["./config/env.yaml"],
    overrides=[
        'grab.subjects=["s1","s2","s3","s4","s5","s6","s7","s8"]',
        'grab.downsample="10fps"',
        'grab.split="train"',
        'grab.use_decimated_obj_meshes=False',
        'grab.generate_obj_keypoints=True',
    ],
)
cfg = init_preprocessing(args)
print("subjects:", list(cfg.grab.subjects))
print("downsample:", cfg.grab.downsample)
print("split:", cfg.grab.split)
print("use_decimated_obj_meshes:", cfg.grab.use_decimated_obj_meshes)
print("generate_obj_keypoints:", cfg.grab.generate_obj_keypoints)
print("input_type:", cfg.input_type)
print("root:", cfg.grab.root)
print("target:", cfg.grab.target)
print("objects_path:", cfg.grab.objects_path)
print("env.datasets_folder:", cfg.env.datasets_folder)
print("env.smpl_folder:", cfg.env.smpl_folder)
