import math
import csv
import random
import numpy as np
from collections import namedtuple

# Dictionary with (playlistID : list of tracks in the playlist) pairs
library = {}
playlists = []
train = []

csvLib = csv.reader(open('../data/playlists_final.csv'), delimiter='\t')
next(csvLib)  # hop header line

for line in csvLib:
    library[line[1]] = []
    playlists.append(line[1])

csvreader = csv.reader(open('../data/train_final.csv'), delimiter='\t')
next(csvreader)  # hop header line

TrainLine = namedtuple("TrainLine", "playlist track")

for line in csvreader:
    library[line[0]].append(line[1])
    train.append(TrainLine(line[0], line[1]))

# format of tenTrLib: list of named tuples Playlist(playlistID, [tracks])
tenTrLib = []
Playlist = namedtuple('Playlist', 'playlist_id tracks')

for playlist in library.keys():
    if len(library[playlist]) >= 10:
        tenTrLib.append(Playlist(playlist, library[playlist]))

tenTrLibTmp = tenTrLib.copy()
thirtyPercent = int(math.floor(len(tenTrLib)/3))
folds = []

randomPlIndices1 = random.sample(range(len(tenTrLibTmp)), thirtyPercent)
tenTrLib1 = [tenTrLibTmp[i] for i in randomPlIndices1]
for index in sorted(randomPlIndices1, reverse=True):
    del tenTrLibTmp[index]

folds.append(tenTrLib1)
print(len(tenTrLib1))

randomPlIndices2 = random.sample(range(len(tenTrLibTmp)), thirtyPercent)
tenTrLib2 = [tenTrLibTmp[i] for i in randomPlIndices2]
for index in sorted(randomPlIndices2, reverse=True):
    del tenTrLibTmp[index]

folds.append(tenTrLib2)
print(len(tenTrLib2))

tenTrLib3 = tenTrLibTmp

folds.append(tenTrLib3)
print(len(tenTrLib3))

csvLib = csv.reader(open('../data/playlists_final.csv'), delimiter='\t')
next(csvLib) #hop header line

playlistDic = {}
i = 0

for line in csvLib:
    playlistDic[line[1]] = i
    i += 1

tracksMatrix = np.genfromtxt('../data/tracks_final.csv', delimiter='\t', dtype='str')

tracks = tracksMatrix[1:,0].tolist()
tracksDic = {}
i = 0

for line in tracks:
    tracksDic[line] = i
    i += 1

for fold in range(3):
    print('fold: ' + str(fold))
    testWriter = csv.writer(open('dataSets/test' + str(fold) + '.csv', 'w', newline=''))
    testWriter.writerow(('playlist_id', 'track_ids'))
    trainTmp = train.copy()
    targetPl = []
    targetTr = []

    # produces the test data files
    #folds[fold] is a list of Playlists namedtuple (playlist_id, tracks)
    for j in range(len(folds[fold])):
        print('working on pl number ' + str(j))
        testRow = ''
        numOfTracks = len(folds[fold][j].tracks)
        targetPl.append(folds[fold][j].playlist_id)

        for k in range(0, 5):
            randomTrackIndex = random.randint(0, numOfTracks - 1)
            track = folds[fold][j].tracks[randomTrackIndex]
            testRow += track + '\t'
            targetTr.append(track)
            trainTmp.remove(TrainLine(folds[fold][j].playlist_id, track))
            del folds[fold][j].tracks[randomTrackIndex]
            numOfTracks -= 1

        testWriter.writerow((folds[fold][j].playlist_id, testRow))

    # produces the train data files removing the test data from the original one
    trainWriter = csv.writer(
        open('dataSets/rawTrain' + str(fold) + '.csv', 'w', newline=''), delimiter='\t')
    trainWriter.writerow(('playlist_id', 'track'))
    for line in trainTmp:
        trainWriter.writerow((line.playlist, line.track))

    #TRANSLATION
    trainList = []
    for tuple in trainTmp:
        trainList.append([tuple.playlist, tuple.track])

    print("Translating")
    for i in range(len(trainList)):
        trainList[i][0] = playlistDic[trainList[i][0]]
        trainList[i][1] = tracksDic[trainList[i][1]]
    print("Translated")

    # produces the train data files removing the test data from the original one
    trainWriter = csv.writer(
        open('dataSets/transTrain' + str(fold) + '.csv', 'w', newline=''), delimiter='\t')
    trainWriter.writerow(('playlist_id', 'track'))
    for line in trainList:
        trainWriter.writerow((line[0], line[1]))

    # produces target playlists
    plWriter = csv.writer(
        open('dataSets/targPl' + str(fold) + '.csv', 'w', newline=''))
    plWriter.writerow(('playlist_id', ))
    for line in targetPl:
        plWriter.writerow((line,))

    # produces target tracks
    trWriter = csv.writer(
        open('dataSets/targTr' + str(fold) + '.csv', 'w', newline=''))
    trWriter.writerow(('track_id', ))
    targetTr_set = set(targetTr)
    targetTr = list(targetTr_set)
    for line in targetTr:
        trWriter.writerow((line,))






