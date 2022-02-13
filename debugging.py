from utils import *
from model import *
from torch.utils.data import DataLoader

def main():
    btds = BrainTumorDataSet()
    # img, mask = btds.__getitem__(0) # BGR

    # cv2.imshow('img', img)
    # cv2.imshow('mask', mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    train_loader = DataLoader(
        btds,
        batch_size=32,
        num_workers=1,
        shuffle=True
    )

    conv2dblock = Conv2dBlock(3, 64)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape)
        print(target.shape)
        out = conv2dblock(data)
        break
    print(out.shape)
    
if __name__ == '__main__':
    main()