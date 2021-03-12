import torch
import torch.nn as nn


class ResBlock(nn.Module):
	def __init__(self, in_ch, out_ch, identity_down=None, stride=1):
		super(ResBlock, self).__init__()
		self.expansion = 4
		self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)
		self.bn1 = nn.BatchNorm2d(out_ch)
		self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1)
		self.bn2 = nn.BatchNorm2d(out_ch)
		self.conv3 = nn.Conv2d(out_ch, out_channels=out_ch*self.expansion, kernel_size=1, stride=1, padding=0)
		self.bn3 = nn.BatchNorm2d(self.expansion*out_ch)
		self.relu = nn.ReLU()
		self.identity_down = identity_down
	
	def forward(self, x):
		identity = x
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.conv3(x)
		x = self.bn3(x)
		
		if self.identity_down is not None:
			identity = self.identity_down(identity)
		x += identity
		x = self.relu(x)
		return x
	
	
class ResNet(nn.Module):
	def __init__(self, block, layers, img_ch, num_classes):
		super(ResNet, self).__init__()
		