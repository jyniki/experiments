'''
Author: Niki
Date: 2021-01-08 11:00:07
Description: 
'''
def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent
