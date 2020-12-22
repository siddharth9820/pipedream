import torch 

class lenet(torch.nn.Module):
    def __init__(self,num_classes=10):
        super(lenet, self).__init__()
        self.l1 = torch.nn.Sequential(torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),torch.nn.ReLU())
        self.l2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.l3 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),torch.nn.ReLU())
        self.l4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.l5 = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Dropout(),torch.nn.Linear(7 * 7 * 64, 1000),torch.nn.ReLU())
        self.l6 = torch.nn.Sequential(torch.nn.Dropout(),torch.nn.Linear(1000,num_classes))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)