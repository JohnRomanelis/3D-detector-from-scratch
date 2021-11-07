import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
from collections import OrderedDict

def print_complex_dict_last(dic):

    for k, v in dic.items():
        if isinstance(v, dict):
            print(k)
            print_complex_dict_last(v)
        else: 
            print(f'{k}: {v[-1]}')

class Tracker(object):
    """ 
        Tracker to track the training process.
        
    Args:
         - loss_keys: keys (names) of the losses that are being tracked
         - acc_metric: if not NONE, should be a list of the names of the accuracy metrics 
                       that we want to track, 
                       if NONE, it will not keep track of any metrics
         - use_tensorboard: (Not Implemented Yet) using tensor board

    """
    def __init__(self, loss_keys, track_metrics=True, use_tensorboard=False):

        self._losses = OrderedDict() #{}
        for k in loss_keys:
            self._losses[k]=[]

        if track_metrics:
            self._metrics = OrderedDict() #{}
            self._metrics.update({
                    '70 70 70': {
                        'mAPbbox':{
                            'easy':[],
                            'moderate':[],
                            'hard':[]
                        },
                        'mAPbev':{
                            'easy':[],
                            'moderate':[],
                            'hard':[]
                        },
                        'mAP3d':{
                            'easy':[],
                            'moderate':[],
                            'hard':[]
                        }
                    },
                    '70 50 50': {
                        'mAPbbox':{
                            'easy':[],
                            'moderate':[],
                            'hard':[]
                        },
                        'mAPbev':{
                            'easy':[],
                            'moderate':[],
                            'hard':[]
                        },
                        'mAP3d':{
                            'easy':[],
                            'moderate':[],
                            'hard':[]
                        }
                    },
                })
        
    def start_new_epoch(self):
        # adding a new registry for every loss of
        # the tracker

        self.counts = 0
        for k in self._losses.keys():
            self._losses[k].append(0)

    def update_losses(self, new_loss):
        # add the new loss to the tracker
        
        # increasing the number of batches 
        self.counts += 1
        
        for k in new_loss.keys():
            self._losses[k][-1] += new_loss[k]
    
    def regularize_epoch_loss(self):

        for k in self._losses.keys():
            self._losses[k][-1] /= self.counts


    def print_losses(self):
        pprint(self._losses)

    def print_latest_losses(self):
        for k in self._losses.keys():
            print(f"{k}: {self._losses[k][-1]}")
        
    @property
    def losses(self):
        return self._losses

    @losses.setter
    def losses(self, data):
        #TODO search for key inconsistancies
        self._losses = data

    def visualize_losses(self, loss_name="all", figs_per_row=3):
        plt.rcParams['axes.grid'] = True
        # ploting all losses
        if loss_name == "all":
            loss_name = self._losses.keys()
        # in one loss represent it as a list with one element
        elif not isinstance(loss_name, list):
            loss_name = [loss_name]


        num_plots = len(loss_name)
        
        # plotting only one loss
        if num_plots == 1:
            fig, ax = plt.subplots()
            k = loss_name[0]
            ax.plot(self._losses[k])
            ax.set_title(k)
        
        # plotting less-equal losses that figs_per_row
        elif num_plots <= figs_per_row:
            fig, ax = plt.subplots(1, num_plots)
                    #sharex=True, sharey=True)
            for i, k in enumerate(loss_name):
                ax[i].plot(self._losses[k])
                ax[i].set_title(k)

        # plotting less losses that figs_per_row
        else:
            num_rows = num_plots // figs_per_row
            if num_plots % figs_per_row > 0:
                num_rows+=1

            fig, ax = plt.subplots(num_rows, figs_per_row)
                    #sharex=True, sharey=True)

            for i, k in enumerate(loss_name):
                r = i // figs_per_row
                c = i % figs_per_row
                ax[r, c].plot(self._losses[k])
                ax[r, c].set_title(k)

        
        fig.set_size_inches(16, 10, forward=True)
        plt.show()


    def add_metrics(self, new_metrics):

        for k in new_metrics.keys():

            for sk in new_metrics[k]:

                for ssk in new_metrics[k][sk]:

                    self._metrics[k][sk][ssk].append(new_metrics[k][sk][ssk]) 

                
    def print_latest_metrics(self):
        print_complex_dict_last(self._metrics)
    
    def print_latest_3d_prec(self):
        easy = self._metrics['70 70 70']['mAP3d']['easy'][-1]
        moderate = self._metrics['70 70 70']['mAP3d']['moderate'][-1]
        hard = self._metrics['70 70 70']['mAP3d']['hard'][-1]

        print(f'{easy} {moderate} {hard}')

    
    def visualize_metrics(self, total_epochs):
        
        easy = self._metrics['70 70 70']['mAP3d']['easy']
        moderate = self._metrics['70 70 70']['mAP3d']['moderate']
        hard = self._metrics['70 70 70']['mAP3d']['hard']

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
        
        x = np.linspace(1, total_epochs, len(easy))

        ax1.plot(x, easy)
        #ax1.set_yticks(np.linspace(0, 100, 10))
        #ax1.grid()
        ax1.set_title("easy")
        ax2.plot(x, moderate)
        #ax2.grid()
        ax2.set_title("moderate")
        ax3.plot(x, hard)
        #ax3.grid()
        ax3.set_title("hard")

        fig.set_size_inches(16, 5, forward=True)

        plt.show()


    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        """
        - metrics: should be a complex dictionary containing lists of metric values
        """
        self._metrics = metrics
    
