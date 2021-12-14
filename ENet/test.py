import torch


class Test:
    """
    Perform testing.
    """

    def __init__(self, model, data_loader, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_once(self, print_loss=False):
        """
        Run an epoch of test.
        ``print_loss`` provide an choice whether printing loss or not.
        Here we calculate the average loss by dividing the length of dataloader instead
            of the length of dataset to represent it is a 'batch loss'.
        """
        self.model.eval()
        epoch_loss = .0
        self.metric.reset()
        for idx, (data, label) in enumerate(self.data_loader):
            data, label = data.to(self.device), label.to(self.device)
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, label)
            epoch_loss += loss.item()

            # Keep track of evaluation the metric
            self.metric.add(output.detach(), label.detach())

            if print_loss:
                print("[Step: %d] Iteration loss: %.4f" % (idx, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
