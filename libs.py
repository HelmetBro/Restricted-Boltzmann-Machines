import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torchvision.utils import make_grid
import torch


def show_and_save(img, file_name):
    r"""Show and save the image.

    Args:
        img (Tensor): The image.
        file_name (Str): The destination.

    """
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imshow(npimg, cmap='gray')
    plt.imsave(f, npimg)


def train(model, device, train_loader, testloader, n_epochs=20, lr=0.01):
    r"""Train a RBM model.

    Args:
        model: The model.
        train_loader (DataLoader): The data loader.
        n_epochs (int, optional): The number of epochs. Defaults to 20.
        lr (Float, optional): The learning rate. Defaults to 0.01.

    Returns:
        The trained model.

    """
    # optimizer
    train_op = optim.Adam(model.parameters(), lr)

    # train the RBM model
    model.train()

    asr_arr = []
    loss_arr = []

    for epoch in range(n_epochs):
        loss_ = []
        for _, (data, target) in enumerate(train_loader):
            v, v_gibbs = model(data.view(-1, 784).to(device))
            loss = model.free_energy(v) - model.free_energy(v_gibbs)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()

        print('Epoch %d\t Loss=%.4f' % (epoch, np.mean(loss_)))
        asr_arr.append(average_stocastic_reconstruction(model, device, testloader))
        loss_arr.append(np.mean(loss_))

    return model, loss_arr, asr_arr

def PCDtrain(model, device, train_loader, testloader, n_epochs=20, lr=0.01):
    r"""Train a RBM model.

    Args:
        model: The model.
        train_loader (DataLoader): The data loader.
        n_epochs (int, optional): The number of epochs. Defaults to 20.
        lr (Float, optional): The learning rate. Defaults to 0.01.

    Returns:
        The trained model.

    """
    # optimizer
    train_op = optim.Adam(model.parameters(), lr)

    # train the RBM model
    model.train()

    asr_arr = []
    loss_arr = []

    for epoch in range(n_epochs):
        loss_ = []
        for i, (data, target) in enumerate(train_loader):
            if i == 0:
                with torch.no_grad():
                    v_gibbs = data.view(-1, 784).to(device)
            with torch.no_grad():
                v, v_gibbs = model(v_gibbs)
            loss = model.free_energy(v) - model.free_energy(v_gibbs)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()

        print('Epoch %d\t Loss=%.4f' % (epoch, np.mean(loss_)))
        asr_arr.append(average_stocastic_reconstruction(model, device, testloader))
        loss_arr.append(np.mean(loss_))

    return model, loss_arr, asr_arr

def average_stocastic_reconstruction(model, device, val_loader):
    model.eval()
    loss = []
    for _, (data, target) in enumerate(val_loader):
        v = data.view(-1, 784).to(device)
        h = model.visible_to_hidden(v)
        v_gibbs = model.hidden_to_visible(h)
        loss.append(torch.dist(v, v_gibbs).item())

    print(f'average_stocastic_reconstruction: {np.mean(loss)}')
    return np.mean(loss)

def plot_images(model, device, val_loader, batch_size):
    model.eval()
    v = next(iter(val_loader))[0].view(-1, 784).to(device)
    h = model.visible_to_hidden(v)
    v_gibbs = model.hidden_to_visible(h)
    # plot real images
    v = v.cpu()
    v_gibbs = v_gibbs.cpu()
    show_and_save(make_grid(v.view(batch_size, 1, 28, 28)), 'output/real')
    plt.show()
    # show the generated images
    show_and_save(make_grid(v_gibbs.view(batch_size, 1, 28, 28).data), 'output/fake')
    plt.show()