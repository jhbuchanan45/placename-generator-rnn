import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

matplotlib.use('Qt5Agg')

model_name = "model-placenames-gb-1631391477"

def create_acc_loss_graph(model_name):
    contents = open("model.log").read().split('\n')

    times = []
    losses = []

    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, loss, val_loss = c.split(",")

            times.append(float(timestamp))
            losses.append(float(loss))

            val_losses.append(float(val_loss))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))


    ax1.plot(times, losses, label="train_loss")
    ax1.plot(times, val_losses, label="test_loss")
    ax1.legend(loc=2)

    plt.show()


create_acc_loss_graph(model_name)
