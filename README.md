# u-net

# how to use
```
usage: train.py [-h] [--train_path TRAIN_PATH] [--valid_path VALID_PATH]    
                [--test_path TEST_PATH] [--rot_degree ROT_DEGREE]    
                [--v_flip_prob V_FLIP_PROB] [--h_flip_prob H_FLIP_PROB]     
                [--distortion_scale DISTORTION_SCALE]    
                [--distortion_prob DISTORTION_PROB]    
                [--in_channels IN_CHANNELS] [--out_channels OUT_CHANNELS]   
                [--lr LR] [--max_epochs MAX_EPOCHS] [--batch_size BATCH_SIZE]   
                [--ds_num_workers DS_NUM_WORKERS] [--model_name MODEL_NAME]    
                [--checkpoints CHECKPOINTS]     
```                 
                data reading
                train_path: training dataset
                          : directory format - you should have the following directories.
                          : TRAIN_PATH/images, TRAIN_PATH/masks
                valid_path: validation dataset
                          : directory format - you should have the following directories.
                          : VALID_PATH/images, VALID_PATH/masks
                test_path : test dataset
                          : yet tobe used.
                
                data augmentation
                rot_degree: degree of RandomRotation
                v_flip_prob: probability of RandomVerticalFlip
                h_flip_prob: probability of RandomHorizontalFlip
                distortion_scale: distortion scale of RandomPerspective
                distortion_prob: probability of distortion
                
                model config
                in_channels: channel of an input image (default= 1)
                out_channels: channel of an output image (default= 1)
                lr: learning rate
                
