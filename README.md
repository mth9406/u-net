# u-net

# how to use
```
(1) train.py
usage: train.py [-h] [--train_path TRAIN_PATH] [--valid_path VALID_PATH]
                [--test_path TEST_PATH] [--rot_degree ROT_DEGREE]
                [--v_flip_prob V_FLIP_PROB] [--h_flip_prob H_FLIP_PROB]
                [--distortion_scale DISTORTION_SCALE]
                [--distortion_prob DISTORTION_PROB] [--model_type MODEL_TYPE]
                [--in_channels IN_CHANNELS] [--out_channels OUT_CHANNELS]
                [--lr LR] [--max_epochs MAX_EPOCHS] [--batch_size BATCH_SIZE]
                [--ds_num_workers DS_NUM_WORKERS] [--patience PATIENCE]
                [--checkpoints CHECKPOINTS]

optional arguments:
  -h, --help            show this help message and exit
  --train_path TRAIN_PATH
  --valid_path VALID_PATH
  --test_path TEST_PATH
  --rot_degree ROT_DEGREE
                        degree of RandomRotation
  --v_flip_prob V_FLIP_PROB
                        probability of RandomVerticalFlip
  --h_flip_prob H_FLIP_PROB
                        probability of RandomHorizontalFlip
  --distortion_scale DISTORTION_SCALE
                        distortion scale of RandomPerspective
  --distortion_prob DISTORTION_PROB
  --model_type MODEL_TYPE
                        model type is either 0 for 'u-net' or 1 for 'deep-u-
                        net' and 2 for 'resnet+u-net'
  --in_channels IN_CHANNELS
  --out_channels OUT_CHANNELS
  --lr LR
  --max_epochs MAX_EPOCHS
  --batch_size BATCH_SIZE
  --ds_num_workers DS_NUM_WORKERS
                        the number of workers in a Dataloader
  --patience PATIENCE   patience of EarlyStopping
  --checkpoints CHECKPOINTS


(2) train_resume.py   
usage: train_resume.py [-h] --model_path MODEL_PATH
                       [--checkpoints CHECKPOINTS] [--model_type MODEL_TYPE]
                       [--train_path TRAIN_PATH] [--valid_path VALID_PATH]
                       [--test_path TEST_PATH] [--rot_degree ROT_DEGREE]
                       [--v_flip_prob V_FLIP_PROB] [--h_flip_prob H_FLIP_PROB]
                       [--distortion_scale DISTORTION_SCALE]
                       [--distortion_prob DISTORTION_PROB]
                       [--in_channels IN_CHANNELS]
                       [--out_channels OUT_CHANNELS] [--lr LR]
                       [--max_epochs MAX_EPOCHS] [--batch_size BATCH_SIZE]
                       [--ds_num_workers DS_NUM_WORKERS] [--patience PATIENCE]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        a path to model (.ckpt format)
  --checkpoints CHECKPOINTS
                        a path to checkpoint
  --model_type MODEL_TYPE
                        model type is either 0 for 'u-net' or 1 for 'deep-u-
                        net'
  --train_path TRAIN_PATH
  --valid_path VALID_PATH
  --test_path TEST_PATH
  --rot_degree ROT_DEGREE
                        degree of RandomRotation
  --v_flip_prob V_FLIP_PROB
                        probability of RandomVerticalFlip
  --h_flip_prob H_FLIP_PROB
                        probability of RandomHorizontalFlip
  --distortion_scale DISTORTION_SCALE
                        distortion scale of RandomPerspective
  --distortion_prob DISTORTION_PROB
                        probability of RandomPerspective
  --in_channels IN_CHANNELS
  --out_channels OUT_CHANNELS
  --lr LR
  --max_epochs MAX_EPOCHS
  --batch_size BATCH_SIZE
  --ds_num_workers DS_NUM_WORKERS
                        the number of workers in a Dataloader
  --patience PATIENCE   patience of EarlyStopping
  
(3) inference.py 
usage: inference.py [-h] --model_path MODEL_PATH [--model_type MODEL_TYPE]
                    [--test_path TEST_PATH]
                    [--prediction_path PREDICTION_PATH] [--thr THR]
                    [--batch_size BATCH_SIZE] [--in_channels IN_CHANNELS]
                    [--out_channels OUT_CHANNELS]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        a path to model (.ckpt format)
  --model_type MODEL_TYPE
                        model type is either 0 for 'u-net' or 1 for 'deep-u-
                        net'
  --test_path TEST_PATH
                        a path to test dataset
  --prediction_path PREDICTION_PATH
                        a path to save predictions
  --thr THR             threshold to generate a mask
  --batch_size BATCH_SIZE
  --in_channels IN_CHANNELS
  --out_channels OUT_CHANNELS
```
