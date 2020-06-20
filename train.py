import csv
import torch
import matplotlib.pyplot as plt
from tcn import TCN
from math import floor


# T,TD,P0,U,DD,FF,WW,c,VV
def load_data(file):
    data = []

    with open("data/%s" % file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)

        for row in reader:
            data.append([float(val) for val in row[1:]])

    return data


input_time = 336
output_time = 24
eval_data = load_data("eval.csv")
train_data = load_data("train.csv")

plt.ion()
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TCN(9, [9])
loss_func = torch.nn.MSELoss().to(device)
optimiser = torch.optim.AdamW(model.parameters())

min_loss = float("inf")
epoch = 0
avg_losses = []
while epoch <= 10:
    model.train()
    for i in range(floor(len(train_data) / input_time)):
        x = torch.tensor(train_data[i:i + input_time]).unsqueeze(2)
        y = torch.tensor(train_data[i + input_time:i + input_time*2]).unsqueeze(2)

        optimiser.zero_grad()
        loss = loss_func(model(x), y)
        loss.backward()
        optimiser.step()

    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(floor(len(eval_data) / input_time)):
            x = torch.tensor(eval_data[i:i + input_time]).unsqueeze(2)
            y = torch.tensor(eval_data[i + input_time:i + input_time*2]).unsqueeze(2)

            losses.append(loss_func(model(x)[:output_time], y[:output_time]))

    avg_loss = sum(losses) / len(losses)
    print(avg_loss)
    avg_losses.append(avg_loss)
    """
    if avg_loss < min_loss:
        min_loss = avg_loss
        torch.save(model.state_dict(), "model.pt")
    else:
        break
    """
    plt.plot(avg_losses)
    plt.draw()
    plt.pause(0.001)

    torch.save(model.state_dict(), "model.pt")

    epoch += 1
