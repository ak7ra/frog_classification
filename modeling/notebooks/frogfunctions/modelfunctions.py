import torch
from torch import manual_seed as torch_manual_seed
from torch.cuda import manual_seed_all
from torch.backends import cudnn
import torch.nn.functional as nnf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from timeit import default_timer as timer
from sklearn.manifold import TSNE

def setup_seed(seed):
    torch_manual_seed(seed)
    manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    
def make_tsne(dataloader, model, epoch):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    labels = []
    for batch, (X, y) in enumerate(dataloader):

        images = X.to(device)
        labels += y

        output = model.forward(images)

        current_outputs = output.cpu().detach().numpy()

        try:
            outputs = np.concatenate([outputs, current_outputs], axis=0)
        except NameError:
            outputs = current_outputs
        
    tsne = TSNE(n_components=2).fit_transform(outputs)

    TSNE_2D(tsne, epoch, labels, scale=True)
            
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
 
def TSNE_2D(tsne, epoch, labels, scale=True):
    
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    if scale:
        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)
    
    colors_per_class = {0:0, 1 : 1, 2: 2, 3:3, 4:4, 5:5, 6:6}
    

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    
    class_to_name = {0: 'Upper Amazon tree frog', 
                                  1: 'Demerara Falls tree frog',
                                  2: 'Chirping Robber frog',
                                  3: "Vanzolini's Amazon frog",
                                  4: "South American common toad",
                                  5: "Peters' dwarf frog",
                                  6: 'Background'}

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        
        indices = [i for i, l in enumerate(labels) if l == label]

        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        ax.scatter(current_tx, current_ty, label=class_to_name[label]) 

    ax.legend(loc='best')
    
    fig.savefig("tsne_epoch"+str(epoch)+".png")
    
    # finally, show the plot
    plt.show()
    
def train_loop(dataloader, model, loss_fn, optimizer):
    # Define device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print current loss 
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        # Correct predictions
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    accuracy = correct / size

    # Print training error
    print(f"Training Error: Accuracy: {(100*accuracy):>0.1f}%")
    
    # Return training accuracy
    return accuracy

def test_loop(dataloader, model, loss_fn, method, epoch, tsne=False):
    # Define device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            # Compute prediction
            pred = model(X)
            
            # Testing loss
            test_loss += loss_fn(pred, y).item()
            
            # Correct predictions 
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # Get probability values 
            prob = nnf.softmax(pred, dim=1)
            probs.append(pd.DataFrame(prob.cpu()))

    if tsne:
        make_tsne(dataloader, model, epoch)
            
    # Test loss
    test_loss /= num_batches
    
    # Test accuracy
    accuracy = correct / size
    
    # Get final probabilities
    probs = pd.concat(probs).reset_index(drop=True)
    probs = probs.rename(columns={0: 'Upper Amazon tree frog', 
                                  1: 'Demerara Falls tree frog',
                                  2: 'Chirping Robber frog',
                                  3: "Vanzolini's Amazon frog",
                                  4: "South American common toad",
                                  5: "Peters' dwarf frog",
                                  6: 'Background'})
    
    if method=='val':
        print(f"Validation Error: Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        # Return accuracy
        return accuracy
    elif method=='test':
        print(f"Test Error: Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    elif method=='prob':
        return probs

def train_model(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, tsne=False):
    train_accuracies = [] 
    val_accuracies = []  

    # Loop
    start = timer() 
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
    
        # Training
        model.train()
        train_accuracy = train_loop(train_dataloader, model, loss_fn, optimizer)
        train_accuracies.append(train_accuracy)
    
        # Validation
        model.eval()
        val_accuracy = test_loop(val_dataloader, model, loss_fn, method='val', epoch=t+1, tsne=tsne)
        val_accuracies.append(val_accuracy)
        
    end = timer()
    print(f'Processing time: {end-start:.2f}s.') 

    # Show performance
    plt.plot(train_accuracies, 'bo-', label="Train accuracy")
    plt.plot(val_accuracies, 'r^-', label="Validation accuracy")
    plt.xticks(range(0,epochs))
    plt.title("Model Performance", fontsize=12)
    plt.legend(loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    
def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def plot_probabilities(predictions, title, xlabel):
    mpl.rcParams['figure.dpi'] = 800
  
    fig, ax = plt.subplots(figsize=(10, 2))
    plt.imshow(predictions.T, cmap='Greens', interpolation = 'nearest', origin='upper')
  
    plt.title(title)

    from matplotlib import ticker
    positions = [0, 1, 2, 3, 4, 5, 6]
    labels = ['Upper Amazon tree frog',
              'Demerara Falls tree frog',
              'Chirping Robber frog',
              "Vanzolini's Amazon frog",
              'South American common toad',
              "Peters' dwarf frog",
              "Background"]

    #highlight_cell(0, 5, color="red", linewidth=1.5)
    #highlight_cell(1, 1, color="red", linewidth=1.5)
    #highlight_cell(2, 4, color="red", linewidth=1.5)

    ax.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    #plt.xticks(np.arange(-0.5, 60.5, 5), np.arange(0, 61, 5))
    plt.xlabel(xlabel)
    plt.tight_layout()
    ax.set_aspect('auto')
    plt.colorbar()
    #plt.savefig('fig1.png', bbox_inches='tight')