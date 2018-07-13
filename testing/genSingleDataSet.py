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

thirtyPercent = int(math.floor(len(tenTrLib)*0.3))


testWriter = csv.writer(open('dataSets/test.csv', 'w', newline=''))
testWriter.writerow(('playlist_id', 'track_ids'))
targetPl = []
targetTr = []

# produces the test data files
for j in range(thirtyPercent):
    randomPlaylistIndex = random.randint(0, thirtyPercent - 1)
    print(str(j))
    testRow = ''
    numOfTracks = len(tenTrLib[randomPlaylistIndex].tracks)
    targetPl.append(tenTrLib[randomPlaylistIndex].playlist_id)

    for k in range(0, 5):
        randomTrackIndex = random.randint(0, numOfTracks - 1)
        track = tenTrLib[randomPlaylistIndex].tracks[randomTrackIndex]
        testRow += track + '\t'
        targetTr.append(track)
        train.remove(TrainLine(tenTrLib[randomPlaylistIndex].playlist_id, track))
        del tenTrLib[randomPlaylistIndex].tracks[randomTrackIndex]
        numOfTracks -= 1

    testWriter.writerow((tenTrLib[randomPlaylistIndex].playlist_id, testRow))
    del tenTrLib[randomPlaylistIndex]
print('songs extracted')

# produces the train data files removing the test data from the original one
trainWriter = csv.writer(
    open('dataSets/rawTrain.csv', 'w', newline=''), delimiter='\t')
trainWriter.writerow(('playlist_id', 'track'))
for line in train:
    trainWriter.writerow((line.playlist, line.track))
print('train file generated')

#TRANSLATION
trainList = []
for tuple in train:
    trainList.append([tuple.playlist, tuple.track])

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

print("Translating")
for i in range(len(trainList)):
    trainList[i][0] = playlistDic[trainList[i][0]]
    trainList[i][1] = tracksDic[trainList[i][1]]
print("Translated")

# produces the train data files removing the test data from the original one
trainWriter = csv.writer(
    open('dataSets/transTrain.csv', 'w', newline=''), delimiter='\t')
trainWriter.writerow(('playlist_id', 'track'))
for line in trainList:
    trainWriter.writerow((line[0], line[1]))
print('train file generated')

# produces target playlists
plWriter = csv.writer(
    open('dataSets/targPl.csv', 'w', newline=''))
plWriter.writerow(('playlist_id', ))
for line in targetPl:
    plWriter.writerow((line,))
print('target pl generated')

# produces target tracks
trWriter = csv.writer(
    open('dataSets/targTr.csv', 'w', newline=''))
trWriter.writerow(('track_id', ))
targetTr_set = set(targetTr)
targetTr = list(targetTr_set)
for line in targetTr:
    trWriter.writerow((line,))
print('target tracks generated')





