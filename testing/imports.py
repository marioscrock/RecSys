import csv

def importRawTrainSet(number):
    """:parameter a two digit string representing the number of the set"""
    """:returns the train set as a list of lists [[playlist-id, track-id]]"""
    result = []
    reader = csv.reader(open('dataSets/rawTrain' + number + '.csv'), delimiter='\t')
    next(reader)  # hop header line
    for line in reader:
        result.append([line[0], line[1]])
    return result

def importTrainSet(number):
    """:parameter a two digit string representing the number of the set"""
    """:returns the train set as a list of lists [[playlist-id, track-id]]"""
    result = []
    reader = csv.reader(open('dataSets/transTrain' + number + '.csv'), delimiter='\t')
    next(reader)  # hop header line
    for line in reader:
        result.append([line[0], line[1]])
    return result

def importTestSet(number):
    """:parameter a two digit string representing the number of the set"""
    """:returns the test set as a list of lists [[playlist-id, [track-id]]]"""
    result = {}
    reader = csv.reader(open('dataSets/test' + number + '.csv'), delimiter=',')
    next(reader)  # hop header line
    for line in reader:
        result = {line[0], line[1].split("\t")[:-1]}
    return result

def importSubmission():
    """:returns the submission as a list of lists [[playlist-id, [track-id]]]"""
    result = {}
    reader = csv.reader(open('../../submission.csv'), delimiter=',')
    next(reader)  # hop header line
    for line in reader:
        result = {line[0], line[1].split("\t")[:-1]}
    return result
