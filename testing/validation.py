import numpy as np
import csv
from collections import namedtuple

class checkTestSet():

    def __init__(self, testpath, trainpath, plpath, trpath):
        targPl = csv.reader(open(plpath))
        next(targPl)  # hop header line

        self.targPl = []

        for line in targPl:
            self.targPl.append(line[0])

        targTr = csv.reader(open(trpath))
        next(targTr)

        self.targTr = []

        for line in targTr:
            self.targTr.append(line[0])

        test = csv.reader(open(testpath), delimiter = ",")
        next(test)  # hop header line

        self.test = {}

        for line in test:
            self.test[line[0]] = [i for i in line[1].split("\t") if len(i) > 0]

        train = csv.reader(open(trainpath), delimiter = "\t")
        next(train)

        self.trainTuples = []
        self.trainStrings = []
        TrainLine = namedtuple("TrainLine", "playlist track")

        for line in train:
            self.trainTuples.append(TrainLine(line[0], line[1]))
            self.trainStrings.append(str(line[0]) + "," + str(line[1]))
        self.trainStrings = np.array(self.trainStrings)

    def checkSongsinTarget(self):
        testTracks = self.test.values()
        testTracks = [item for sublist in testTracks for item in sublist]
        for song in self.targTr:
            if not (song in testTracks):
                print("Error! " + song)
                exit()
        testTracks = np.array(list(set(testTracks)))
        targTr = np.array(self.targTr)
        diff = np.setdiff1d(testTracks, targTr)
        if len(diff) == 0:
            print("Wasssgoooood")
        else:
            print('Error! size of diff: ' + str(len(diff)))



    def checkPlaylistinTarget(self):
        testPlaylists = list(self.test.keys())
        for playlist in self.targPl:
            if not (playlist in testPlaylists):
                print("Error! " + playlist)
                exit()
        print("Wasssgoooood")

    def checkSongsNotinTrain(self):
        testStrings = []
        for playlist in self.test.keys():
            for song in self.test[playlist]:
                testStrings.append(playlist + "," + song)
        testStrings = np.array(testStrings)
        print(testStrings)
        print(self.trainStrings)
        check = np.in1d(testStrings, self.trainStrings)
        if np.any(check):
            print(len(testStrings[check]))
            print(len(testStrings))
            exit()
        print("Wasssgoooood")



