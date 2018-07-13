import numpy as np
from scipy import sparse
from saveload import *
from similarity import *

def buildSIM(ICM = None, URM = None, algorithm = 'CF', simFunction = 'dotProduct', shkg = 0):

    if algorithm == 'CB':

        if ICM == None:
            print("You have to provide an ICM matrix!")
            return

        matrix = ICM.T

    elif algorithm == 'CF':

        if URM == None:
            print("You have to provide an URM matrix!")
            return

        matrix = URM.T

    if simFunction == 'dotProduct':
        return matrix * matrix.T

    elif simFunction == 'cosine':
        dist = Cosine(shrinkage=shkg)
        return dist.compute(matrix)

    elif simFunction == 'adjustedCosine':
        dist = AdjustedCosine(shrinkage=shkg)
        return dist.compute(matrix)

    elif simFunction == 'jaccard':
        #to be implemented
        print("Not yet implemented")

    elif simFunction == 'pearson':
        dist = Pearson(shrinkage=shkg)
        return dist.compute(matrix)
