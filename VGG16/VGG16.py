import torch
import torch.nn as nn

VGGNET = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG(nn.Module):
	def __init__(self, in_ch, num_classes):
		super(VGG, self).__init__()
		self.in_ch = in_ch
		self.conv_layers = self.generate_cnn(VGGNET)
		self.fcs = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, num_classes)
		)
	
	def forward(self, x):
		x = self.conv_layers(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fcs(x)
		return x
	
	def generate_cnn(self, config):
		layers = []
		in_ch = self.in_ch
		
		for layer in config:
			if type(layer) == int:
				out_ch = layer
				
				layers += [
					nn.Conv2d(
						in_channels=in_ch, out_channels=out_ch,
						kernel_size=(3, 3), stride=(1, 1),
						padding=(1, 1)
					),
					nn.BatchNorm2d(layer),
					nn.ReLU()
				]
				in_ch = out_ch
			elif layer == 'M':
				layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
		
		return nn.Sequential(*layers)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGG(in_ch=3, num_classes=1000).to(device)
X = torch.randn(1, 3, 224, 224).to(device)
print(model(X).shape)
