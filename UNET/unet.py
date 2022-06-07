import torch
import torch.nn as nn
import torchvision.transforms.functional as f


class DoubleConv(nn.Module):
	"""
	Build for U-net block
	"""
	def __init__(self, in_ch, out_ch):
		super(DoubleConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)
	
	def forward(self, x):
		x = self.conv(x)
		return x


class UNET(nn.Module):
	"""
	Build for U-net Network
	"""
	def __init__(self, in_ch, out_ch=1, features=[64, 128, 256, 512]):
		super(UNET, self).__init__()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		
		for feature in features:
			self.downs.append(DoubleConv(in_ch, feature))
			in_ch = feature
		
		for feature in features[::-1]:
			self.ups.append(
				nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
			)
			self.ups.append(DoubleConv(feature * 2, feature))
		
		self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
		self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)
	
	def forward(self, x):
		skip_connections = []
		for downs in self.downs:
			x = downs(x)
			skip_connections.append(x)
			x = self.pool(x)
		x = self.bottleneck(x)
		skip_connections = skip_connections[::-1]
		for idx in range(0, len(self.ups), 2):
			x = self.ups[idx](x)
			skip_conn = skip_connections[idx//2]
			if x.shape != skip_conn.shape:
				x = f.resize(x, size=skip_conn.shape[2:])
			concat = torch.cat((skip_conn, x), dim=1)
			x = self.ups[idx+1](concat)
		
		x = self.final_conv(x)
		return x


if __name__ == '__main__':
	x = torch.randn((3, 1, 160, 160))
	model = UNET(in_ch=1, out_ch=1)
	pred = model(x)
	print(x.shape)
	print(pred.shape)
	assert pred.shape == x.shape
