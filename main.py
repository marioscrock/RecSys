from testing.metrics import *
from URMbuilder import buildURM
from SIMbuilder import buildSIM
from ICMbuilder import buildICM
from toRec import *
from scipy import sparse as sps
import numpy as np

def testing():

    train_file = 'testing/dataSets/transTrain.csv'
    t_pl_file = 'testing/dataSets/targPl.csv'
    t_tr_file = 'testing/dataSets/targTr.csv'

    recs = recommend(train_file, t_pl_file, t_tr_file)
    metrics = evaluateSubmission('testing/dataSets/test.csv', recs)

    print("Recommender performance is: Precision = {:.8f}, Recall = {:.8f}, MAP = {:.8f}".format(
        metrics['precision'], metrics['recall'], metrics['MAP']))

    return metrics

def testing_3fold():

    metrics = []

    for i in range(3):
        train_file = 'testing/dataSets/transTrain' + str(i) + '.csv'
        t_pl_file = 'testing/dataSets/targPl' + str(i) + '.csv'
        t_tr_file = 'testing/dataSets/targTr' + str(i) + '.csv'
        recs = recommend(train_file, t_pl_file, t_tr_file)
        metrics.append(evaluateSubmission('testing/dataSets/test' + str(i) + '.csv', recs))

    precision = 0
    recall = 0
    MAP = 0

    for i in range(len(metrics)):
        precision += metrics[i]['precision']
        recall += metrics[i]['recall']
        MAP += metrics[i]['MAP']

    for i in range(len(metrics)):
        print("FOLD" + str(i) + " metrics are: Precision = {:.8f}, Recall = {:.8f}, MAP = {:.8f}".format(
            metrics[i]['precision'], metrics[i]['recall'], metrics[i]['MAP']))

    print('')
    print("Recommender performance is: Precision = {:.8f}, Recall = {:.8f}, MAP = {:.8f}".format(
        precision/len(metrics), recall/len(metrics), MAP/len(metrics)))


def submission():

    train_file = 'data/trainIndex.csv'
    t_pl_file = 'data/target_playlists.csv'
    t_tr_file = 'data/target_tracks.csv'

    recs = recommend(train_file, t_pl_file, t_tr_file)

    tplaylist = []
    csvTPlaylist = csv.reader(open(t_pl_file))
    next(csvTPlaylist)  # hop header line

    for line in csvTPlaylist:
        tplaylist.append(line[0])

    subwriter = csv.writer(open('submission.csv', 'w', newline=''))
    subwriter.writerow(('playlist_id', 'track_ids'))

    for i in range(len(tplaylist)):

        string = ''
        for k in range(5):
            string += recs[tplaylist[i]][k] + '\t'

        subwriter.writerow((tplaylist[i], string))


def recommend(train_file, t_pl_file, t_tr_file):

    # buildICM(artists = True, albums = True, tags = True, idf = True, svd = False)
    ICM = buildICM(albums = False, tags = False, idf = True)
    ICM2 = buildICM(artists = False, tags = False, idf = True)

    # buildURM(train_file, idf = False, playcountNorm = False)
    URM = buildURM(train_file)
    URMidf = buildURM(train_file, idf = True)

    # buildSIM(ICM = None, URM = None, algorithm = 'CF', simFunction = 'dotProduct', shkg = 0)
    SIMCB1 = buildSIM(ICM=ICM, algorithm='CB', simFunction='adjustedCosine')
    SIMCB2 = buildSIM(ICM=ICM2, algorithm='CB', simFunction='adjustedCosine')
    SIMCF = buildSIM(URM=URMidf, algorithm='CF', simFunction='pearson')
    SIMCF = reduceKNN(SIMCF, 2200)
    SIMUU = buildSIM(URM=URMidf.T, algorithm='CF', simFunction='pearson')
    SIMUU = reduceKNN(SIMUU, 75)

    # normalize(simURM, path=None) - If path != None saves the matrix
    simURMCBArtists = normalize(URMidf * SIMCB1)
    simURMCBAlbums = normalize(URMidf * SIMCB2)
    simURMCF = normalize(URMidf * SIMCF)
    simURMUU = normalize(SIMUU * URMidf)

    #Hybrid simURM
    simURM = 0.16 * (simURMCBArtists + 0.6 * simURMCBAlbums) + 1 * simURMCF + 1.1 * simURMUU

    #Recommend  5 songs for playlist penalyzing after each round of recommendations
    #already recommended songs with weights given as parameter
    return toRecSimUrmPenalyze(t_pl_file, t_tr_file, URM, simURM, [0.9, 0.88, 0.88, 0.88])


#Command to generate recommendation for submission()
submission()
