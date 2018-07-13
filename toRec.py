import csv
from scipy import sparse
from saveload import load_sparse_csr,save_sparse_csr
from attributes import *

#Selects 5 recommended tracks in t_tr_file for playlists in t_pl_file
#through ratings and a penalization policy in user rating prediction matrix.

#The algorithm at each iteration for each playlist selects the track not already in the playlist
#with the highest score in the matrix for the given playlist.
#After each iteration the algorithms penalyzes columns of scores related to the set of tracks
#recommended in the current iteration using the weight related.

#simURMs = user rating prediction matrix
#pen_weights = penalyzation weights for each iteration
#topPop = how many top popular item to exclude from recs
def toRecSimUrmPenalyze(t_pl_file, t_tr_file, URM, simURM, pen_weights, topPop = 0):

    playlists = []

    csvLib = csv.reader(open('data/playlists_final.csv'), delimiter='\t')
    next(csvLib) #hop header line

    for line in csvLib:
        playlists.append(line[1])

    tracksMatrix = np.genfromtxt('data/tracks_final.csv', delimiter='\t', dtype='str')
    tracks = tracksMatrix[1:,0].tolist()

    ttrack = []

    csvTTracks = csv.reader(open(t_tr_file))
    next(csvTTracks) #hop header line

    for line in csvTTracks:
        ttrack.append(line[0])

    tplaylist = []

    csvTPlaylist = csv.reader(open(t_pl_file))
    next(csvTPlaylist) #hop header line

    for line in csvTPlaylist:
        tplaylist.append(line[0])

    submission = {}

    mask = np.in1d(np.array(tracks), np.array(ttrack), assume_unique=True)

    if topPop > 0:
        URMi = np.array(URM.sum(axis=0)).squeeze()
        maskTopPop = URMi < (np.sort(URMi)[-topPop])


    for j in range(5):

        tracks_to_pen = []
        print('Computing recommendations')

        for i in range(len(playlists)):
            # Take for each playlist in target_playlist the correspondent row
            if playlists[i] in tplaylist:

                if j == 0:
                    submission[playlists[i]] = []

                u = np.ravel(simURM[i].todense())
                zeros = np.zeros(1)
                seen = np.in1d(np.ravel(URM[i].todense()), zeros)
                mytrack = np.array(tracks)

                andMask = np.logical_and(mask, seen)

                for k in submission[playlists[i]]:
                    andMask[tracks.index(k)] = False

                if topPop > 0:
                    andMask = np.logical_and(andMask, maskTopPop)

                u = u[andMask]
                mytrack = mytrack[andMask]
                sortedIndex = list(np.argsort(u))

                recs = mytrack[sortedIndex[-1]]
                tracks_to_pen.append(tracks.index(recs))

                submission[playlists[i]].append(recs)

        tracks_to_pen_set = np.unique(np.array(tracks_to_pen))

        if j < 4 :
            print('Penalizing ' + str(len(tracks_to_pen_set)) + ' tracks')
            simURM.data[np.in1d(simURM.indices, tracks_to_pen_set)] *= pen_weights[j]

        print('Recommended ' + str(j+1) + ' items')

    return submission

#Selects recommended track in t_tr_file for playlists in t_pl_file
#doing a weighted average of ratings in user rating prediction matrices in simURMs list
#simURMs = list of simURM matrices to be weighted in the hybrid
#weights = weights related to matrices in simURMs (same order required)
#n_rec = number of tracks to recommend for each playlist
#topPop = how many top popular item to exclude from recs
#tracks_rec = list of tuples (path to file containing items to penalyze, weight for those items)
def toRecSimUrm(t_pl_file, t_tr_file, URM, simURMs, weights=[1], n_rec = 5, topPop = 0, tracks_rec = None):

    playlists = []

    csvLib = csv.reader(open('data/playlists_final.csv'), delimiter='\t')
    next(csvLib) #hop header line

    for line in csvLib:
        playlists.append(line[1])

    tracksMatrix = np.genfromtxt('data/tracks_final.csv', delimiter='\t', dtype='str')
    tracks = tracksMatrix[1:,0].tolist()


    ttrack = []

    csvTTracks = csv.reader(open(t_tr_file))
    next(csvTTracks) #hop header line

    for line in csvTTracks:
        ttrack.append(line[0])

    tplaylist = []

    csvTPlaylist = csv.reader(open(t_pl_file))
    next(csvTPlaylist) #hop header line

    for line in csvTPlaylist:
        tplaylist.append(line[0])

    submission = {}

    mask = np.in1d(np.array(tracks), np.array(ttrack), assume_unique=True)

    if topPop > 0:
        URMi = np.array(URM.sum(axis=0)).squeeze()
        maskTopPop = URMi < (np.sort(URMi)[-topPop])

    if tracks_rec != None:

        csvreader = csv.reader(open(tracks_rec[0]), delimiter=' ')
        next(csvreader)

        indices = []
        for line in csvreader:
            k = tracks.index(line[0])
            indices.append(k)

    for i in range(len(playlists)):
    #Take for each playlist in target_playlist the correspondent row
        if playlists[i] in tplaylist:

            zeros = np.zeros(1)
            seen = np.in1d(np.ravel(URM[i].todense()), zeros)
            mytrack = np.array(tracks)

            andMask = np.logical_and(mask, seen)

            if topPop > 0:
                andMask = np.logical_and(andMask, maskTopPop)

            u = np.ravel(simURMs[0][i].todense())[andMask] * weights[0]

            if tracks_rec != None:
                u = tracks_rec_weight(indices, tracks_rec[1], u)

            if weights != None:
                for t in range(1, len(simURMs)):
                    uAdd = np.ravel(simURMs[t][i].todense())[andMask]
                    u += uAdd * weights[t]

            mytrack = mytrack[andMask]

            sortedIndex = list(np.argsort(u))

            recs = []
            for k in range(1, n_rec + 1 ):
                recs.append(mytrack[sortedIndex[-k]])

            print(str(i) + " " + str(playlists[i]) + ". " + str(recs))
            submission[playlists[i]] = recs

    return submission

def reduceKNN(SIM, KNN):

    SIMnew = sparse.lil_matrix((SIM.shape[0], SIM.shape[1]))
    print('computing KNN')
    for i in range(SIM.shape[0]):
        s = np.ravel(SIM[i].todense())
        indexSorted = np.argsort(s)
        np.put(s, indexSorted[:-KNN], 0)
        SIMnew[i] = sparse.lil_matrix(s)

    SIMnew = SIMnew.tocsr()
    SIMnew.eliminate_zeros()

    return SIMnew

#Matrix normalized on maximum value of the matrix
def normalize(simURM, path=None):

    max1 = max(simURM.data)
    print("Max found " + str(max1))
    simURM = np.divide(simURM, max1)
    simURM = simURM.tocsr()

    if path != None:
        save_sparse_csr(path, simURM)

    print("simURM normalized")

    return simURM

#Each row of the matrix is normalized on its own maximum value
def normalize2(simURM, path=None):

    print('normalizing rows')

    for i in range(simURM.shape[0]):
        if simURM.data[simURM.indptr[i]:simURM.indptr[i + 1]].shape[0] != 0:
            row_max = np.max(simURM.data[simURM.indptr[i]:simURM.indptr[i + 1]])
            simURM.data[simURM.indptr[i]:simURM.indptr[i + 1]] /= row_max

    if path != None:
        save_sparse_csr(path, simURM)

    print("simURM normalized")

    return simURM

def tracks_rec_weight(tracks_rec_indices, value, s):

    for index in tracks_rec_indices:
        s[index] *= value

    return s

def createFile_recTracks(rec, path = 'data/tracks_rec.csv'):

    tracks_rec = []

    for playlist in rec.keys():
        tracks_rec += rec[playlist]

    f = open(path, 'w', newline='')
    subwriter = csv.writer(f)
    subwriter.writerow(('track_id',))

    tracks_rec_set = set(tracks_rec)

    for track in tracks_rec_set:
        subwriter.writerow((track,))

    f.close()

def createFile_recSubmission(recs, t_pl_file, path = 'data/submission.csv'):

    tplaylist = []
    csvTPlaylist = csv.reader(open(t_pl_file))
    next(csvTPlaylist)  # hop header line

    for line in csvTPlaylist:
        tplaylist.append(line[0])

    subwriter = csv.writer(open(path, 'w', newline=''))
    subwriter.writerow(('playlist_id', 'track_ids'))

    n_rec = len(recs[tplaylist[0]])

    for i in range(len(tplaylist)):

        string = ''
        for k in range(n_rec):
            string += recs[tplaylist[i]][k] + '\t'

        subwriter.writerow((tplaylist[i], string))
