import numpy as np
import csv
from scipy import sparse
from math import log

# Import tracks data from file
def tracksGetter():
    tracks = np.genfromtxt('data/tracks_final.csv', delimiter='\t', dtype='str')
    return tracks

# Extract distinct artists IDs in <artists> list
def distinctArtists(tracks):
    artists = tracks[1:, 1].tolist()
    artist_id_set = set(artists)
    artists = list(artist_id_set)

    return artists

# Extract distinct albums names in <album> list
def distinctAlbums(tracks):
    album = tracks[1:, 4].tolist()
    album_set = set(album)
    album = list(album_set)
    album = [i[1:-1] for i in album] # list comprehension to delete "[" and "]" chars
    album.remove('None') # removes useless NULL attribute
    album.remove('')

    return album

# Extract distinct tags in <tags> list
def distinctTags(tracks):
    l_tag = tracks[1:, 5].tolist()

    tag_set = set(l_tag)
    l_tag = list(tag_set)
    # Returns list of lists of tags
    lol_tags = map(lambda x: [i for i in x[1:-1].split(', ') if len(i)>0], l_tag)
    # Flattens the list of lists
    tags = [item for sublist in lol_tags for item in sublist]
    distinct_tags = set(tags)
    tags = list(distinct_tags)

    return tags



