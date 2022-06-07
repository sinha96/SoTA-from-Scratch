import torch
import torch.nn as nn


class ResBlock(nn.Module):
	"""
	Build for Resnet block
	"""
	def __init__(self, in_ch, out_ch, identity_down=None, stride=1):
		super(ResBlock, self).__init__()
		self.expansion = 4
		self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)
		self.bn1 = nn.BatchNorm2d(out_ch)
		self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1)
		self.bn2 = nn.BatchNorm2d(out_ch)
		self.conv3 = nn.Conv2d(out_ch, out_channels=out_ch * self.expansion, kernel_size=1, stride=1, padding=0)
		self.bn3 = nn.BatchNorm2d(self.expansion * out_ch)
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
	"""
	build for Residual network Model
	"""
	def __init__(self, block, layers, img_ch, num_classes):
		super(ResNet, self).__init__()
		self.in_ch = 64
		self.conv1 = nn.Conv2d(img_ch, 64, kernel_size=7, stride=2, padding=3)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
		self.layer1 = self._make_layers(block, layers[0], out_ch=64, stride=1)
		self.layer2 = self._make_layers(block, layers[1], out_ch=128, stride=2)
		self.layer3 = self._make_layers(block, layers[2], out_ch=256, stride=2)
		self.layer4 = self._make_layers(block, layers[3], out_ch=512, stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * 4, num_classes)
	
	def _make_layers(self, block, num_resblock, out_ch, stride):
		identity_down = None
		layers = []
		
		if self.in_ch != out_ch * 4 or stride != 1:
			identity_down = nn.Sequential(
				nn.Conv2d(self.in_ch, out_ch * 4, kernel_size=1, stride=stride),
				nn.BatchNorm2d(out_ch * 4)
			)
		layers.append(ResBlock(self.in_ch, out_ch, identity_down, stride))
		self.in_ch = out_ch * 4
		for i in range(num_resblock - 1):
			layers.append(ResBlock(self.in_ch, out_ch))
		return nn.Sequential(*layers)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		
		x = self.avgpool(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc(x)
		return x

# Functions for building different version of resnet network


def resnet50(in_ch=3, num_classes=1000):
	return ResNet(ResBlock, [3, 4, 6, 3], in_ch, num_classes)


def resnet101(in_ch=3, num_classes=1000):
	return ResNet(ResBlock, [3, 4, 23, 3], in_ch, num_classes)


def resnet152(in_ch=3, num_classes=1000):
	return ResNet(ResBlock, [3, 8, 36, 3], in_ch, num_classes)


def run():
	net50 = resnet50()
	net101 = resnet101()
	net152 = resnet152()
	x = torch.randn(64, 3, 224, 224)
	y_50 = net50(x).to('cuda')
	print(y_50.shape)
	del y_50
	y_101 = net101(x).to('cuda')
	print(y_101.shape)
	del y_101
	y_152 = net152(x).to('cuda')
	print(y_152.shape)
	del y_152


if __name__ == '__main__':
	run()
