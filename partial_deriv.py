# -*- coding: utf-8 -*-

import numpy as np


def partial_deriv(m,X,Y):
    sum = 0
    for i in range(0,m):
        sum += (h[X[i]]-Y[i])*X[i]
    return sum/m
