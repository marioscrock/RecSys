import csv
import numpy as np
from scipy import sparse
from math import log
import operator
import random
from saveload import *

# submitted = list of dictionaries (playlist:recommended_items) OR list of submission files
def ranking(submitted, mode = 'roundRobin'):

    submission = []

    for i in range(len(submitted)):

        if type(submitted[i]) is dict:
            submission.append(submitted[i])
        else:
            reader = csv.reader(open(submitted[i]), delimiter=',')
            next(reader)  # hop header line

            sub = {}

            for line in reader:
                sub[line[0]] = [i for i in line[1].split('\t') if len(i) > 0]

            submission.append(sub)

    if mode == 'roundRobin':

        ranking = submission[0].copy()

        for key in ranking.keys():
            ranking[key] = []

        for playlist in ranking.keys():

            k = 0
            positions = [0 for i in range(len(submission))]

            while (k < 5):

                index = positions.index(min(positions))

                r = submission[index][playlist][positions[index]]

                if r not in ranking[playlist]:
                    ranking[playlist].append(r)
                    k += 1

                positions[index] += 1

        return ranking

    if mode == 'averageRanking':

        ranking = submission[0].copy()

        for key in ranking.keys():
            ranking[key] = []

        for playlist in ranking.keys():

            tracks = {}

            for d in range(len(submission)):
                for track in submission[d][playlist]:
                    tracks[track] = []

            for d in range(len(submission)):
                for track in submission[d][playlist]:
                    tracks[track].append(submission[d][playlist].index(track))

            for track in tracks.keys():
                tracks[track] = sum(list(map(lambda x:x*x,tracks[track]))) / len(tracks[track]) - len(tracks[track]) * 60

            print(tracks)

            for k in range(5):
                r = min(tracks.items(), key=operator.itemgetter(1))[0]
                ranking[playlist].append(r)
                tracks[r] = 1000

        print(ranking)
        return ranking














