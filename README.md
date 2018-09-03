# Recommender Systems | Challenge 2017 #

Repository for the project/challenge of the **Recommender System** 2017 course at Politecnico di Milano by Mario Scrocca and Moreno Raimondo Vendra.

**Private Kaggle competition**
The application domain is a music streaming service, where users listen to tracks (songs) and create playlists of favorite songs. The main goal of the competition is to discover which track a user will likely add to a playlist, based on:
* other tracks in the same playlist
* other playlists created by the same user

In this competition you are required to predict a list of 5 tracks for a set of playlists. The original unsplitted dataset includes around 1M interactions (tracks belonging to a playlist) for 57k playlists and 100k items (tracks). A subset of about 10k playlists and 32k items has been selected as test playlists and items. The goal is to recommend a list of 5 relevant items for each playlist. MAP@5 is used for evaluation. 

**Team name**: Juventus & Peroni

**Notes** 
* Datasets not included in this repo.
* ```similarity.py``` code partially taken from repository https://github.com/MaurizioFD/RecSys_Course_2017

### Algorithms implemented ###
* `v0.1` - **Top popular**
* `v0.2` - Top popular (without tracks already in playlists)
* `v1.0` - **Content Based** Recommender (only Artists and Albums considered as attributes)
* `v2.0` - **Collaborative Filtering** Recommender (user-user and item-item)
* `v2.1` - CB and CBF changing similarities functions (Adjusted-Cosine, Pearson)
* `v3.0` - **Weighted Hybrid** of previous recommenders
* `v4.0` - Hybrid of CB (artists and albums), CF (user and item) **at each step penalyzing tracks already recommended**
