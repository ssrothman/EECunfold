import numpy as np

def forward(arrs_gen, transfer):
    gen = arrs_gen['gen']
    covgen = arrs_gen['covgen']

    forward = transfer @ gen
    covforward = transfer @ covgen @ transfer.T

    return forward, covforward
