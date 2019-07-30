
import torch
import torch.nn as nn

class ExampleNet(nn.Module):

	def __init__(self):
		super(ExampleNet, self).__init__()
		self.linear_3 = nn.Linear(in_features = 28*28, out_features = 128, bias = True)
		self.reLU_5 = nn.ReLU(inplace = False)
		self.linear_6 = nn.Linear(in_features = 128, out_features = 256, bias = True)
		self.reLU_7 = nn.ReLU(inplace = False)
		self.linear_9 = nn.Linear(in_features = 256, out_features = 300, bias = True)
		self.reLU_12 = nn.ReLU(inplace = False)
		self.linear_10 = nn.Linear(in_features = 300, out_features = 512, bias = True)
		self.reLU_11 = nn.ReLU(inplace = False)
		self.linear_8 = nn.Linear(in_features = 512, out_features = 10, bias = True)

	def forward(self, x_para_1):
		x_reshape_4 = torch.reshape(x_para_1,shape = (-1,28*28))
		x_linear_3 = self.linear_3(x_reshape_4)
		x_reLU_5 = self.reLU_5(x_linear_3)
		x_linear_6 = self.linear_6(x_reLU_5)
		x_reLU_7 = self.reLU_7(x_linear_6)
		x_linear_9 = self.linear_9(x_reLU_7)
		x_reLU_12 = self.reLU_12(x_linear_9)
		x_linear_10 = self.linear_10(x_reLU_12)
		x_reLU_11 = self.reLU_11(x_linear_10)
		x_linear_8 = self.linear_8(x_reLU_11)
		return x_linear_8
