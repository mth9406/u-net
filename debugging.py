from utils import *
from model import *
from torch.utils.data import DataLoader

def main():
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')
    
    print(device)
    
    btds = BrainTumorDataSet('./data/train')
    # img, mask = btds.__getitem__(0) # BGR

    # cv2.imshow('img', img)
    # cv2.imshow('mask', mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    train_loader = DataLoader(
        btds,
        batch_size=4,
        num_workers=1,
        shuffle=True
    )

    # model = DoubleConv(3, 32).to(device)
    model = Unet(1, 1).to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape)
        print(target.shape)
        data = data.to(device)
        out = model(data)
        break
    print(out.shape)
    
if __name__ == '__main__':
    main()