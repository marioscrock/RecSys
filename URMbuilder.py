import numpy as np
import csv
from scipy import sparse
from saveload import *
from math import log

def buildURM(train_file, idf = False, playcountNorm = False, enrichFile = None):

    csvreader = csv.reader(open(train_file), delimiter='\t')
    next(csvreader)

    URM_tuples = []
    numberInteractions = 0

    for line in csvreader:
        URM_tuples.append(tuple(line))
        numberInteractions += 1

    if enrichFile != None:
        csvenrich = csv.reader(open(enrichFile), delimiter='\t')
        next(csvenrich)
        for line in csvenrich:
            URM_tuples.append(tuple(line))
            numberInteractions += 1

    playlistList, trackList = zip(*URM_tuples)

    playlistList = list(playlistList)
    playlistList = list(map(lambda x: int(x), playlistList))
    trackList = list(trackList)
    trackList = list(map(lambda x: int(x), trackList))

    ones = [1 for i in range(0, numberInteractions)]
    URM = sparse.coo_matrix((ones, (playlistList, trackList)))
    print(URM.shape)
    URM = URM.tocsr()

    if idf == True:
        # Build URM cleaned version with TF-IDF (in our case TF = 1)
        nItems = URM.shape[1]
        URMidf = sparse.lil_matrix((URM.shape[0], URM.shape[1]))

        for i in range(0, URM.shape[0]):
            IDF_i = log(nItems/np.sum(URM[i]))
            URMidf[i] = np.multiply(URM[i], IDF_i)
            print(i)

        URM = URMidf.tocsr()

    if playcountNorm == True:

        tracks = np.genfromtxt('data/tracks_final.csv', delimiter='\t', dtype='str')
        playcounts = np.array(tracks[1:, 3])
        playcounts[playcounts == ''] = 0.0
        playcounts = playcounts.astype(float)
        playcounts += 1e-6
        playcounts = np.log(playcounts)

        nUsers = URM.shape[0]
        URM = URM.T
        URM = URM.tocsr()

        for i in range(0, URM.shape[0]):
            sum = np.sum(URM[i])
            IDF_i = log(nUsers / sum)
            URM[i] = np.multiply(URM[i].multiply(sum/playcounts[i]), IDF_i)
            print(i)

        URM = (URM.T).tocsr()

    print("URM READY!")

    return URM

def buildEnrichFile(submitted, num_interactions, output_path):

    #generates an additional train file to use URM builder to enrich the URM efficiently
    reader = csv.reader(open(submitted), delimiter=',')
    next(reader)  # hop header line
    i = 0

    for line in reader:
        recs = [(line[0], i) for i in line[1].split('\t') if len(i) > 0]
        # cut first num_interactions recommendations
        recs = recs[:num_interactions]
        i += 1
        print(i)

    with open(output_path, 'w', newline='') as f:
        f.write('playlist_id, track_ids')
        for rec in recs:
            f.write(str(rec[0]) + "\t" + str(rec[1]))

    print('URM enriched with ' + str(len(recs)) + ' interactions')

    return output_path
