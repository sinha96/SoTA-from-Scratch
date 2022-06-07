import torch
import torch.nn as nn
from math import ceil

base_model = [
	# expansion_rh, channels, repeats, stride, kernel_size
	[1, 16, 1, 1, 3],
	[6, 24, 2, 2, 3],
	[6, 40, 2, 2, 3],
	[6, 88, 3, 2, 3],
	[6, 112, 3, 1, 5],
	[6, 192, 4, 2, 5],
	[6, 320, 1, 1, 3],
]

phi_val = {
	# phi_val, res, drop_rate
	'b0': (0, 224, 0.2),
	'b1': (0.5, 240, 0.2),
	'b2': (1, 260, 0.3),
	'b3': (2, 300, 0.3),
	'b4': (3, 380, 0.4),
	'b5': (4, 456, 0.4),
	'b6': (5, 528, 0.5),
	'b7': (6, 600, 0.5)
}


class ConvBlock(nn.Module):
	"""
	Block of convolution network of efficient network

	"""
	def __init__(self, in_ch, out_ch, kernel_size, stride, padding, groups=1):
		super(ConvBlock, self).__init__()
		self.conv = nn.Conv2d(
			in_ch, out_ch, kernel_size,
			stride, padding, groups=groups,
			bias=False
		)
		self.bn = nn.BatchNorm2d(out_ch)
		self.silu = nn.SiLU()
	
	def forward(self, x):
		"""
		Propagating incoming feature in the network
		:param x: input feature
		:return: output of the network
		"""
		x = self.conv(x)
		x = self.bn(x)
		x = self.silu(x)
		
		return x


class SqueezeExcitation(nn.Module):
	"""
	Excitation block of Efficient network
	"""
	def __init__(self, in_ch, reduce_dim):
		super(SqueezeExcitation, self).__init__()
		self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(in_ch, reduce_dim, 1),
			nn.SiLU(),
			nn.Conv2d(reduce_dim, in_ch, 1),
			nn.Sigmoid(),
		)
	
	def forward(self, x):
		"""
		Propagating incoming feature in the network
		:param x: incoming feature
		:return: output of the network
		"""
		x *= self.se(x)
		return x


class InvertedResidualBlock(nn.Module):
	"""
	Residual Block of the efficient model
	"""
	def __init__(self, in_ch, out_ch, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):
		super(InvertedResidualBlock, self).__init__()
		self.survival_prob = survival_prob
		self.use_residual = in_ch == out_ch and stride == 1
		hidden_dim = in_ch * expand_ratio
		self.expand = in_ch != hidden_dim
		reduced_dim = int(in_ch / reduction)
		
		if self.expand:
			self.expand_conv = ConvBlock(in_ch, hidden_dim, kernel_size=3, stride=1, padding=1)
		
		self.conv = nn.Sequential(
			ConvBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
			SqueezeExcitation(hidden_dim, reduced_dim),
			nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
			nn.BatchNorm2d(out_ch)
		)
	
	def _stochastic_depth(self, x):
		if not self.training:
			return x
		binary_tensor = torch.randn(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
		
		x = torch.div(x, self.survival_prob)
		x *= binary_tensor
		return x
	
	def forward(self, ip):
		x = self.expand_conv(ip) if self.expand else ip
		if self.use_residual:
			x = self.conv(x)
			x = self._stochastic_depth(x) + ip
			return x
		else:
			x = self.conv(x)
			return x


class EfficientNet(nn.Module):
	"""
	Efficient Network Model
	"""
	def __init__(self, version, num_class):
		super(EfficientNet, self).__init__()
		width_factor, depth_factor, dropout_rate = self.calculate_factor(version)
		last_ch = ceil(1280 * width_factor)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.features = self.create_feature(width_factor, depth_factor, last_ch)
		self.classifier = nn.Sequential(
			nn.Dropout(dropout_rate),
			nn.Linear(last_ch, num_class)
		)
	
	def calculate_factor(self, version, alpha=1.2, beta=1.1):
		phi, res, drop_rate = phi_val[version]
		depth_factor = alpha ** phi
		width_factor = beta ** phi
		return width_factor, depth_factor, drop_rate
	
	def create_feature(self, width_fac, depth_fac, last_ch):
		ch = int(32 * width_fac)
		features = [ConvBlock(3, ch, 3, stride=2, padding=1)]
		in_ch = ch
		
		for expand_ratio, ch, repeats, stride, kernel_size in base_model:
			out_ch = 4 * ceil(int(ch * width_fac) / 4)
			layers_repeats = ceil(repeats * depth_fac)
			for layer in range(layers_repeats):
				features.append(
					InvertedResidualBlock(
						in_ch, out_ch, expand_ratio=expand_ratio, stride=stride if layer == 0 else 1,
						kernel_size=kernel_size, padding=kernel_size // 2
					)
				)
				in_ch = out_ch
		
		features.append(
			ConvBlock(in_ch, last_ch, kernel_size=1, stride=1, padding=0)
		)
		return nn.Sequential(*features)
	
	def forward(self, x):
		x = self.features(x)
		x = self.pool(x)
		x = x.view(x.shape[0], -1)
		x = self.classifier(x)
		return x


if '__main__' == __name__:
	device = "cuda" if torch.cuda.is_available() else "cpu"
	version = "b0"
	phi, res, drop_rate = phi_val[version]
	num_examples, num_classes = 4, 10
	x = torch.randn((num_examples, 3, res, res)).to(device)
	model = EfficientNet(
		version=version,
		num_class=num_classes,
	).to(device)
	
	print(model(x).shape)  # (num_examples, num_classes)


