class Train:
    """
    Perform training.
    Using CrossEntropy loss to directly perform optimization.
    """

    def __init__(self, model, data_loader, optim, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_once(self, print_loss=False):
        """
        Runs an epoch of training
        ``print_loss`` provide an choice whether printing loss or not.
        Here we calculate the average loss by dividing the length of dataloader instead
            of the length of dataset to represent it is a 'batch loss'.
        """
        self.model.train()
        epoch_loss = .0
        self.metric.reset()
        for idx, (data, label) in enumerate(self.data_loader):
            data, label = data.to(self.device), label.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            epoch_loss += loss.item()

            #
            self.metric.add(output.detach(), label.detach())
            if print_loss:
                print("[Index: %d] Iteration loss: %.4f" % (idx, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
