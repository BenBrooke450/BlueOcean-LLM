
from importlib.metadata import version

print("torch version:", version("torch"))

import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


input_vector = inputs[1]
print(input_vector)
#tensor([0.5500, 0.8700, 0.6600])


input_1 = inputs[0]
print(input_1)
#tensor([0.4300, 0.1500, 0.8900])

print(0.55*0.43 + 0.87*0.15 + 0.66*0.89)
#0.9544

print(torch.dot(input_vector,input_1))
#tensor(0.9544)







print(inputs.shape[0])
#6

query = inputs[1]  # 2nd input token is the query
print(query)
#tensor([0.5500, 0.8700, 0.6600])

attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)
#tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])



attn_scores_2_temp = attn_scores_2 / attn_scores_2.sum()

print(attn_scores_2_temp)
#tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])


def sofrmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

print(sofrmax(attn_scores_2))
#tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])


print(torch.softmax(attn_scores_2,dim = 0))
#tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])














