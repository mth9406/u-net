import torch

if __name__=='__main__':
    print(torch.cuda.is_available())
    print(torch.__version__)