# 1. Nearest neighbour

# Write a function that gets a training set and a test sample and returns the label of the training point
# the closest to the latter.
# More precisely, write:
# def nearest˙classification(train˙input, train˙target, x):

# where
# • train˙input is a 2d float tensor of dimension n × d containing the training vectors,
# • train˙target is a 1d long tensor of dimension n containing the training labels,
# • x is 1d float tensor of dimension d containing the test vector,
# and the returned value is the class of the train sample closest to x for the L
# 2 norm.

# Hint: The function should have no python loop, and may use in particular torch.mean , torch.view ,
# torch.pow , torch.sum , and torch.sort or torch.min . My version is 164 characters long.
import torch

train_input = torch.rand(3,5)
train_input2 = torch.rand(3,5)
x = torch.rand(1,5)
train_target = torch.tensor([1,5,3])

# def nearestClassification(train_input, train_target, x):
torch.manual_seed(4)
# difference between matrix and vector
z = train_input - x
# squared
z2 = torch.pow(train_input - x,2)
#sum
z3 = torch.sum(z2,1)
#square root
z4 = torch.sqrt(z3)
#min
min = torch.min(z4,0)
index = min[1]
print(train_target[index])



