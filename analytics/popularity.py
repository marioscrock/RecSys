import matplotlib.pyplot as pyplot
from attributes import *
import numpy as np

class AnalyticsPlotter():
    def __init__(self):
        self.tracks  = np.genfromtxt('../data/tracks_final.csv', delimiter='\t', dtype='str')
        self.artists = distinctArtists(self.tracks)
        self.albums = distinctAlbums(self.tracks)
        self.tags = distinctTags(self.tracks)
        self.playcounts = np.array(self.tracks[1:, 3])
        self.playcounts[self.playcounts == ''] = 0.0
        self.playcounts = self.playcounts.astype(float)
        self.plTimestamps = []
        csvLib = csv.reader(open('../data/playlists_final.csv'), delimiter='\t')
        next(csvLib)  # hop header line
        for line in csvLib:
            self.plTimestamps.append(line[0])

    def tagsPop(self):
        l_tag = self.tracks[1:, 5].tolist()
        # Returns list of lists of tags
        lol_tags = map(lambda x: [i for i in x[1:-1].split(', ') if len(i) > 0], l_tag)
        # Flattens the list of lists
        all_tags = np.array([int(item) for sublist in lol_tags for item in sublist])
        unique, counts = np.unique(all_tags, return_counts=True)
        counts= np.sort(counts)
        pyplot.plot(counts, "bo")
        pyplot.ylabel('Popularity')
        pyplot.xlabel('Tags')
        pyplot.show()

    def artistPop(self):
        l_art = self.tracks[1:, 1]
        unique, counts = np.unique(l_art, return_counts=True)
        counts= np.sort(counts)
        pyplot.plot(counts, "bo")
        pyplot.ylabel('Popularity')
        pyplot.xlabel('Artists')
        pyplot.show()

    def albumPop(self):
        all_albums = list(self.tracks[1:, 4])
        all_albums = [x[1: -1] for x in all_albums]
        all_albums = [int(i) for i in all_albums if i != 'None' and len(i)>0]
        unique, counts = np.unique(all_albums, return_counts=True)
        counts = np.sort(counts)
        pyplot.plot(counts, "bo")
        pyplot.ylabel('Songs')
        pyplot.xlabel('Albums')
        pyplot.show()

    def showPlaycounts(self):
        self.playcounts += 1e-6
        counts = np.log(self.playcounts)
        counts = np.sort(counts)
        print(str(np.mean(counts)))
        pyplot.plot(counts, "bo")
        pyplot.ylabel('playcounts')
        pyplot.xlabel('Songs')
        pyplot.show()


    def playlistsSize(self):
        train = np.genfromtxt('../data/train_final.csv', dtype='str')
        unique, counts = np.unique(train[1:, 0], return_counts=True)
        counts = np.sort(counts)
        pyplot.plot(counts, "bo")
        pyplot.ylabel('Freq')
        pyplot.xlabel('TrainRows')
        pyplot.show()

    def trainStructure(self):
        train = csv.reader(open('../data/train_final.csv'))
        trainList = []
        for i in train:
            trainList.append(i)
        unique, counts = np.unique(trainList, return_counts=True)
        print(counts)
        mask = counts > 1
        if any(mask):
            print(max(counts))
            exit()

    def timestampsPop(self):
        unique, counts = np.unique(self.plTimestamps, return_counts=True)
        counts = np.sort(counts)
        print('max: ' + str(np.max(counts)))
        print('mean: ' + str(np.mean(counts)))
        print('above avg: ' + str(len(counts[counts > np.mean(counts)])))
        pyplot.plot(counts, "bo")
        pyplot.ylabel('Popularity')
        pyplot.xlabel('CreatedAt')
        pyplot.show()

    def timestamps(self):
        unique, counts = np.unique(self.plTimestamps, return_counts=True)
        unique = np.sort(unique.astype(int))
        print('max: ' + str(np.max(unique)))
        print('min: ' + str(np.min(unique)))
        print('qty: ' + str(len(unique)))
        print('mean: ' + str(np.mean(unique)))
        print('above avg: ' + str(len(unique[unique > np.mean(unique)])))
        pyplot.plot(unique, counts, "bo")
        pyplot.ylabel('Popularity')
        pyplot.xlabel('CreatedAt')
        pyplot.show()

    def binningts(self):
        binnedTs = []
        tsArray = np.array(self.plTimestamps).astype(int)
        min = np.min(tsArray)
        tsArray = tsArray - min
        tsArray = np.sort(tsArray)
        bin = 1
        bin_size = np.max(tsArray)//500
        current_bin = bin_size*bin
        for ts in tsArray:
            while ts > current_bin:
                bin = bin + 1
                current_bin = bin_size * bin
            binnedTs.append(current_bin)
        unique, counts = np.unique(binnedTs, return_counts=True)
        unique = np.sort(unique.astype(int))
        print('bins: ' + str(len(unique)))
        pyplot.plot(unique, counts, "bo")
        pyplot.ylabel('Popularity')
        pyplot.xlabel('CreatedAt')
        pyplot.show()

plotter = AnalyticsPlotter()
plotter.binningts()
