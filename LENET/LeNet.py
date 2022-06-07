import torch
import torch.nn as nn


class LeNet(nn.Module):
	"""
	Build for LeNet Network
	"""
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=0)
		self.relu = nn.ReLU()
		self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=0)
		self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=0)
		self.linear1 = nn.Linear(120, 64)
		self.linear2 = nn.Linear(64, 10)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.pool(x)
		
		x = self.conv2(x)
		x = self.relu(x)
		x = self.pool(x)
		
		x = self.conv3(x)
		x = self.relu(x)
		
		x = x.reshape(x.shape[0], -1)
		
		x = self.linear1(x)
		x = self.relu(x)
		x = self.linear2(x)
		
		return x


X = torch.randn(64, 1, 32, 32)
model = LeNet()
print(model(X).shape)
