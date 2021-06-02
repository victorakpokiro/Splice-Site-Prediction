
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
# from torchvision import datasets
# from torchvision.transforms import ToTensor, Lambda
from sklearn.model_selection import train_test_split

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(400, 4, 9)
		self.conv2 = nn.Conv2d(4, 1, 1)
		self.conv3 = nn.Conv2d(1, 128, 3, padding=1, bias=False)
		self.conv4R = nn.Conv2d(128, 128, 3, padding=1, bias=False)
		self.conv5 = nn.Conv2d(128, 1, 3, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		# He initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		residual = x
		x2 = self.conv3(x)
		out = x2
		for _ in range(25):
			out = self.conv4R(self.relu(self.conv4R(self.relu(out))))
			out = torch.add(out, x2)

		out = self.conv5(self.relu(out))
		out = torch.add(out, residual)
		return out



def load_data():

	labels = np.loadtxt('label.txt')
	encoded_seq = np.loadtxt('encoded_seq.txt')
	
	x_train,x_test,y_train,y_test = train_test_split(encoded_seq,labels,test_size=0.1)

	# xnp_array = np.array(x_train)
	# xtrain_np = torch.from_numpy(xnp_array)

	# ynp_array = np.array(y_train)
	# ytrain_np = torch.from_numpy(ynp_array)

	# xtxnp_array = np.array(x_test)
	# xtest_np = torch.from_numpy(xtxnp_array)

	# ytxnp_array = np.array(y_test)
	# ytest_np = torch.from_numpy(ytxnp_array)

	return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)
	# return xtrain_np, ytrain_np, xtest_np, ytest_np



learning_rate = 1e-3
batch_size = 50
epochs = 5
x_train,y_train,x_test,y_test = load_data()
train_dataloader = DataLoader(x_train, batch_size=batch_size, shuffle=True)

x_train = torch.from_numpy(x_train)
# torch.reshape(x_train, (-1, 400))
x_train = x_train.numpy()
x_train = np.expand_dims(x_train, 1)
x_train = np.expand_dims(x_train, 1)


train_loader = torch.utils.data.DataLoader(data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=True)
# train_features, train_labels = next(iter(train_dataloader))
# test_dataloader = DataLoader(test_data, batch_size=64)


def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	for batch, (X, y) in enumerate(dataloader):
		# Compute prediction and loss
		pred = model(X)
		loss = loss_fn(pred, y)

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def main():
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = Net().to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
	epochs = 10


	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		train(train_loader, model, loss_fn, optimizer)
		# test_loop(test_dataloader, model, loss_fn)
	print("Done!")




if __name__ == '__main__':
	main()
