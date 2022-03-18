#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import numpy as py
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[3]:


sentences = ["i love you","he loves me","she likes basketball","i hate you","sorry for that","that is awful"]
labels = [1,1,1,0,0,0]#"1 is good ,0 is bad"

#TextCNN Parameter
embedding_size = 2 #wordembed dim
sequence_length = len(sentences[0]) #every sentence contains squence_length(=3) words
num_classes = len(set(labels))#0 or 1
print("num_classes:",num_classes)
batch_size = 3

word_list = " ".join(sentences).split()
print("word_list:",word_list)
vocab = list(set(word_list))
print("vocab:",vocab)
word2idx = {w: i for i,w in enumerate(vocab)}
print("word2idx:",word2idx)
vocab_size = len(vocab)
print("vocab_size:",vocab_size)


# In[4]:


targets = []
for out in labels:
    targets.append(out)
print(targets)


# In[5]:


def make_data(sentences,labels):
    inputs = []#tensor[[11, 5, 10], [12, 6, 14], [8, 1, 3], [11, 9, 10], [7, 0, 13], [13, 2, 4]]
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])
        
    targets = []
    for out in labels:
        targets.append(out) #To using Torch Softmax Loss function
    return inputs,targets

input_batch,target_batch = make_data(sentences,labels)
input_batch,target_batch = torch.LongTensor(input_batch),torch.LongTensor(target_batch)

print("input_batch",input_batch,"target_batch",target_batch)
dataset = Data.TensorDataset(input_batch,target_batch)
loader = Data.DataLoader(dataset,batch_size,True)

"""
input_batch tensor([[11,  5, 10],
        [12,  6, 14],
        [ 8,  1,  3],
        [11,  9, 10],
        [ 7,  0, 13],
        [13,  2,  4]]) 
        target_batch tensor([1, 1, 1, 0, 0, 0])
"""
        


# In[6]:


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN,self).__init__()
        self.W = nn.Embedding(vocab_size,embedding_size)#vocab_size:15,embedding_size:2
        #print("W:",self.W)
        output_channel = 3
        self.conv = nn.Sequential(
            # conv : [input_channel(=1),output_channel,(filter_height,filter_width),stride=1]  [,]卷积核大小
            nn.Conv2d(1,output_channel,(2,embedding_size)),
            nn.ReLU(),
            #pool : ((filter_height,filter_width))
            nn.MaxPool2d((2,1)),
        )
        #fc
        self.fc = nn.Linear(output_channel,num_classes)
    
    def forward(self,X):
        """
        x:[batch_size,sequence_length]
        """
        batch_size = X.shape[0]
        #print("X:",X)
        embedding_X = self.W(X) #[batch_size, sequence_length,embedding_size]
        #print("embedding_X:",embedding_X)
        embedding_X = embedding_X.unsqueeze(1)# add channel(=1) [batch, channel(=1),sequence_length,embedding_size]
        #print("embedding_X unqueeze:",embedding_X)  embedding_X: []
        conved = self.conv(embedding_X) # [batch_size,output_channel,1,1]
        flatten = conved.view(batch_size,-1) # [batch_size,output_channel * 1 * 1]
        output = self.fc(flatten)
        return output
        
        


# In[ ]:





# In[7]:


model = TextCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3)

#Train
for epoch in range(5000):
    for batch_x,batch_y in loader:
        batch_x,batch_y = batch_x.to(device),batch_y.to(device)
        #print("batch_x:",batch_x.shape) 3,3
        pred = model(batch_x)
        loss = criterion(pred,batch_y)
        if(epoch + 1) % 1000 == 0:
            print("Epoch:",'%04d' % (epoch + 1),'loss =','{:.6f}'.format(loss))
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        


# In[9]:


# Test
test_text = 'i hate me'
tests = [[word2idx[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests).to(device)

#Predict
model = model.eval()
predict = model(test_batch).data.max(1,keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text,"is Bad Mean")
else :
    print(test_text,"is Good Mean!")


# In[ ]:




