#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import cv2, os, tqdm
from multiprocessing import Process, Queue
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from ubteacher.engine.trainer import *

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

# hacky way to register
from ubteacher.modeling import *
from ubteacher.engine import *
from ubteacher import add_ubteacher_config

import pdb
    

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # custom dataset
    cfg.DATASETS.TRAIN = ("cook_train",)
    cfg.DATASETS.TEST = ("cook_val",)
    #
    cfg.freeze()
    default_setup(cfg, args)
    
    return cfg


def main(args):
    
    # register custom datasets
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("cook_train", {}, "/content/drive/MyDrive/COOK/detection_test/v8/train/_annotations.coco.json", "/content/drive/MyDrive/COOK/detection_test/v8/train")
    register_coco_instances("cook_val", {}, "/content/drive/MyDrive/COOK/detection_test/v8/valid/_annotations.coco.json", "/content/drive/MyDrive/COOK/detection_test/v8/valid")
    register_coco_instances("cook_test", {}, "/content/drive/MyDrive/COOK/detection_test/v8/test/_annotations.coco.json", "/content/drive/MyDrive/COOK/detection_test/v8/test")
    
    # set metadata to dataset
    MetadataCatalog.get("cook_train").set(thing_classes=["foods","bacon_cooked","bacon_overcooked","bacon_raw","egg_cooked","egg_overcooked","egg_raw","others","pan","pancake_cooked","pancake_overcooked","pancake_raw"])

    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
        Trainer = UBRCNNTeacherTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelStudent)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res
    
    if args.test_only:
        # inference only
        # no evaluation
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelStudent)
            # test model class for using run_on_video method below
            # test_model = TestModel(Trainer, cfg, ensem_ts_model.modelTeacher)
        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)

            # test_model = TestModel(Trainer, cfg, model)
        test_model = CustomPredictor(cfg)

        # frame detection
        global v
        v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

        assert os.path.isfile(args.video_input)
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output_dir:
            if os.path.isdir(args.output_dir):
                output_fname = os.path.join(args.output_dir, basename)
                
                from datetime import datetime
                now = datetime.now()
                output_fname = os.path.splitext(output_fname)[0] + "_output_"+ now.astimezone().strftime('%Y-%m-%d %H:%M:%S') +".mp4"
            else:
                output_fname = args.output_dir + "output.mp4"
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
            print("video detection starts...")
            readFrames, maxFrame = 0, args.max_frame
            assert maxFrame >= 0, "maxFrame should be over 0"

            while(video.isOpened()):
                ret, frame = video.read()

                if not ret:
                    break
                
                outputs = test_model(frame)

                # Make sure the frame is colored
                # pdb.set_trace()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Draw a visualization of the predictions using the video visualizer
                visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))

                # Convert Matplotlib RGB format to OpenCV BGR format
                visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

                readFrames += 1
                if maxFrame and readFrames > maxFrame: #최대 프레임 수 조절
                    break

                cv2.imwrite('test_img.png', visualization)
                output_file.write(visualization)
                # else:
                #     cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                #     cv2.imshow(basename, frame)
                #     if cv2.waitKey(1) == 27:
                #         break  # esc to quit
            video.release()
            output_file.release()

            return

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--test-only", action="store_true", help="perform test only")
    parser.add_argument("--max_frame", type=int, default=0)
    parser.add_argument("--output_dir")
    parser.add_argument("--video_input")
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
