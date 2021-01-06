from tflearn.callbacks import Callback

# define the early-stop callback
class EarlyStoppingCallback(Callback):
    def __init__(self, val_loss_thresh, val_loss_patience):
        """ minimum loss improvement setup """
        self.val_loss_thresh = val_loss_thresh
        self.val_loss_last = float('inf')
        self.val_loss_patience = val_loss_patience
        # number of iters validation test failed since last success
        self.val_loss_num_failed = 0

    def on_batch_end(self, training_state, snapshot=False):
        """ loss improvement threshold w/ patience """
        # Apparently this can happen.
        if training_state.val_loss is None: return

        if (self.val_loss_last
          - training_state.val_loss) < self.val_loss_thresh:
          # failed too many times, exit.
          if self.val_loss_num_failed >= self.val_loss_patience:
            raise StopIteration
          else:
            self.val_loss_num_failed += 1
        else:
            # loss good again - reset num_failed to 0
            # record validation loss
          self.val_loss_last = training_state.val_loss
          self.val_loss_num_failed = 0
