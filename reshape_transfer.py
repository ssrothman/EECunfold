import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Reshape transfer matrix')
parser.add_argument('folder', type=str)
parser.add_argument('--sepPt', action='store_true')

args = parser.parse_args()

folder = args.folder

if args.sepPt:
    transferPT = np.load(os.path.join(folder, 'transferPT.npy'))
    transferRest = np.load(os.path.join(folder, 'transferRest.npy'))
    print(transferPT.shape)
    print(transferRest.shape)
    print(np.sum(transferRest))
    print()
    eye_order = np.eye(transferRest.shape[0])
    transferPT = np.nan_to_num(transferPT/np.sum(transferPT, axis=0, keepdims=True))
    
    transfer = np.einsum('ajcdkl,ai,bj->abcdijkl', transferRest, eye_order, transferPT, optimize=True)
    print(transfer.shape)
    print(np.sum(transfer))

    np.save(os.path.join(folder, 'transfer_goodshape.npy'), transfer)

else:
    transfer = np.load(os.path.join(folder, 'transfer.npy'))
    print(transfer.shape)
    print(np.sum(transfer))
    print()
    eye_order = np.eye(transfer.shape[0])
    transfer = np.einsum('abcdjkl,ai->abcdijkl', transfer, eye_order, optimize=True)
    print(transfer.shape)
    print(np.sum(transfer))

    np.save(os.path.join(folder, 'transfer_goodshape.npy'), transfer)
