import pickle
import torch 
import numpy as np

class TwoLayerNet(torch.nn.Module):
    def __init__(self,H1):
        super(TwoLayerNet, self).__init__()
        self.control1 = torch.nn.Linear(2,H1)
        self.control2 = torch.nn.Linear(H1,H1)
        self.control3 = torch.nn.Linear(H1,2)

    def forward(self,x):
        h2 = torch.relu(self.control1(x))
        h3 = torch.relu(self.control2(h2))
        u = self.control3(h3)
        return u

if __name__ == "__main__":
    with open('train_data.pickle','rb') as f:
        data_input, data_output = pickle.load(f)
    data_input = np.array(data_input)
    data_input = data_input[:,[1,0]]
    model = TwoLayerNet(32)
    data = torch.FloatTensor(data_input)
    label = torch.FloatTensor(data_output)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.995)
    for i in range(20000):
        out = model(data)
        error = torch.norm(out-label)
        loss = error.mean()
        if i%10 == 0:
            print(i,loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(model(torch.FloatTensor([128, 34])))
    torch.save(model.state_dict(), 'perception_model')