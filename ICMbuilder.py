import numpy as np
from scipy import sparse
from attributes import *
from saveload import *


def buildICM(artists = True, albums = True, tags = True, idf = True):

    tracks = tracksGetter()
    artistsList = distinctArtists(tracks)
    albumsList = distinctAlbums(tracks)
    tagsList = distinctTags(tracks)

    csvreader = csv.reader(open('data/tracksIndex.csv'), delimiter='\t')
    next(csvreader)

    ICM_tuples = []
    numberPairs = 0

    for line in csvreader:
        ICM_tuples.append(tuple(line))
        numberPairs += 1

    trackIdsList, attributesList = zip(*ICM_tuples)

    trackIdsList = list(trackIdsList)
    trackIdsList = list(map(lambda x: int(x), trackIdsList))
    attributesList = list(attributesList)
    attributesList = list(map(lambda x: int(x), attributesList))

    ones = [1 for i in range(0, numberPairs)]
    ICM = sparse.coo_matrix((ones, (attributesList, trackIdsList)))
    print(ICM.shape)
    ICM = ICM.tolil()

    indices = []
    offset = 0
    if artists == False:
        indices += [i for i in range(0, len(artistsList))]
    offset += len(artistsList)
    if albums == False:
        indices += [i for i in range(offset, len(albumsList) + offset)]
    offset += len(albumsList)
    if tags == False:
        indices += [i for i in range(offset, len(tagsList) + offset)]

    indices = np.array(indices)
    ICM.rows = np.delete(ICM.rows, indices)
    ICM.data = np.delete(ICM.data, indices)
    ICM._shape = (ICM._shape[0] - len(indices), ICM._shape[1])
    ICM = ICM.tocsr()

    if idf:
        # Build ICM cleaned version with TF-IDF (in our case TF = 1)
        nItems = ICM.shape[1]
        ICMidf = sparse.lil_matrix((ICM.shape[0], ICM.shape[1]))

        for i in range(0, ICM.shape[0]):
            IDF_i = log(nItems / np.sum(ICM[i]))
            ICMidf[i] = np.multiply(ICM[i], IDF_i)
            print(i)

        ICM = ICMidf.tocsr()

    return ICM

def translateAttributes(artists = True, albums = True, tags = True):

    # Import tracks data from file to build ICM
    tracks = tracksGetter()
    offset = 0

    if artists:
        # Extract distinct artists IDs in <artists> list
        artistsList = distinctArtists(tracks)

        artistsIDs = [i for i in range(offset, len(artistsList))]
        offset += len(artistsList)

    if albums:
        # Extract distinct albums names in <album> list
        albumsList = distinctAlbums(tracks)

        albumsIDs = [i for i in range(offset, len(albumsList) + offset)]
        offset += len(albumsList)


    if tags:
        # Extract distinct tags in <tags> list
        tagsList = distinctTags(tracks)

        tagsIDs = [i for i in range(offset, len(tagsList) + offset)]

    subwriter = csv.writer(open('data/tracksIndex2.csv', 'w', newline=''), delimiter='\t')
    subwriter.writerow(('track_index', 'attribute_Id'))

    for i in range(1, tracks.shape[0]):

        print(i)

        if artists:
            artist = tracks[i, 1]
            if artist in artistsList:
                subwriter.writerow((i-1, artistsIDs[artistsList.index(artist)]))

        if albums:
            album = tracks[i, 4][1:-1]
            if album in albumsList:
                subwriter.writerow((i-1, albumsIDs[albumsList.index(album)]))

        if tags:
            trackTags = [i for i in tracks[i, 5][1:-1].split(', ') if len(i) > 0]
            for tag in trackTags:
                if tag in tagsList:
                    subwriter.writerow((i-1, tagsIDs[tagsList.index(tag)]))

def buildUCM(URM, ICM, ICMContent):
    ICM = ICM.T
    UCM = sparse.lil_matrix((URM.shape[0], ICM.shape[1]))
    print(UCM.shape)
    indices = np.array(range(ICM.shape[1]))

    for i in range(URM.shape[0]):

        iSongs = URM.indices[URM.indptr[i]:URM.indptr[i + 1]]
        row = np.sum(ICM[iSongs], axis=0)
        row = np.asarray(row).ravel()
        if ICMContent == 'Tags':
            mask = row > 0.5*len(iSongs)
        elif ICMContent == 'AlAr':
            mask = row > 2*len(iSongs)
        row_max = np.max(row)
        if row_max > 0:
            row = np.divide(row, row_max)
        ind = indices[mask]
        for j in ind:
            UCM[i, j] = row[j]
        print(str(i))

    return UCM.tocsr().T
