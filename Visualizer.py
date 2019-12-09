import myconfig
import requests
import pandas as pd
import json
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from dateutil import parser

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def get_lastfm_top(user):
    """
    Returns a list of songs and their playcounts that are the top tracks for a given
    user
    """
    url = "https://ws.audioscrobbler.com/2.0/?method=user.gettoptracks&user="+user+"2&api_key="\
        +myconfig.api_key+"&format=json&period=12month&limit=1000"

    r = requests.get(url)
    pages = r.json()['toptracks']['@attr']
    songs = r.json()['toptracks']['track']
    for x in range(2,int(pages['totalPages'])+1):
        url = "https://ws.audioscrobbler.com/2.0/?method=user.gettoptracks&user=naterich2&api_key="\
            +myconfig.api_key+"&format=json&period=12month&limit=1000"
        print("Working on page: "+str(x))
        url = url+"&page="+str(x)
        r = requests.get(url)
        songs.extend(r.json()['toptracks']['track'])
    return songs

def get_top_tags(songs):
    """
    For each song in songs, get a list of 5 top tags that are associated with it on lastfm
        meant to be used in conjunction with get_lastfm_top
    """
    my_dict = {}
    excluded = {}
    my_set = {}
    counter = 0
    for i in range(0,10000):

        if counter % 5 ==0:
            print("Done: "+str(counter))
        artist = urllib.parse.quote(songs[i]['artist']['name'], safe='')
        song = urllib.parse.quote(songs[i]['name'], safe='')
        song_url = "https://ws.audioscrobbler.com/2.0/?method=track.gettoptags&artist="\
            +artist+"&track="+song+"&api_key="+myconfig.api_key+"&format=json"
        song_tag_results = requests.get(song_url)
        if song_tag_results.status_code != 200:
            print(song_tag_results.status_code)
          # if song_tag_results.headers['Content-Length'] > 0
            try:
                print(song_tag_results.json())
            except Exception as e:
                print(e)
            finally:
                print(song_url, song_tag_results.content)
            continue
       #print(song_tag_results.headers)
        if 'toptags' not in song_tag_results.json():
            print(song_url, song_tag_results.json())
            continue
        tags_json = song_tag_results.json()['toptags']['tag']
        tags = []
        if len(tags_json) > 0:
            if len(tags_json) >= 5:
                tags = [a['name'] for a in tags_json[1:5]]
            else:
                tags = [a['name'] for a in tags_json]
            for tag in tags:
                if tag not in my_set:
                    my_set[tag] = 1
                else:
                    my_set[tag]+= 1
            my_dict[artist+"_"+song] = {'name':song,'artist':artist,'playcount': songs[i]['playcount'],'tags':tags}
        else:
            excluded[artist+"_"+song] = {'name':song,'artist':artist}
        counter=counter+1
    return (my_dict,my_set)

def clean_tracks_and_tags(my_dict,tag_thresh,playcount_thresh):
    """
    Takes dict from get_top_tags, a tag threshold and a playcount threshold and generates
    a DataFrame with each song name as an index, and the name, artist, playcount, and all the genres
    as columns.  Writes raw data from my_dict and the dataframe to a csv as output
    """
    top_tags = [a for a in my_set.keys() if my_set[a] > tag_thresh]
#print(my_dict['Snail%20Mail_Heat%20Wave'])
#print({a: 1 for a in my_dict['Snail%20Mail_Heat%20Wave']['tags']})
    df_columns = ['name','artist','playcount']+top_tags

    my_df = pd.DataFrame(columns=df_columns)
    for i, key in enumerate(my_dict.keys()):
        song = my_dict[key]
        if int(song['playcount']) > playcount_thresh:

            tags = {a: 1 for a in song['tags']}
        #print({'name':song['name'], 'artist': song['artist']})
        #print({'name':song['name'], 'artist': song['artist']}.update(tags))

            my_series = pd.Series({**tags, **{'name':song['name'], 'artist': song['artist'],'playcount':song['playcount']}})
            my_df.loc[i] = my_series
    my_df = my_df.fillna(0)
    my_df.to_csv('song_data.csv')
    #print(my_df.columns)
    raw = pd.DataFrame.from_dict(my_dict,orient='index')
    raw.to_csv("song_data_raw.csv")

def generate_lastfm_plots():
    """
    Uses the csv built by clean_tracks_and_tags to make PCA plot, TSNE plot, and a scatter plot
    """
    my_df = pd.read_csv('song_data.csv')
    #print(my_df.drop(['name','artist','playcount','Unnamed: 0'],axis=1))
    pca = PCA(n_components=2)
    pca_reduction = PCA(n_components=10)
    reduction_fit = pca_reduction.fit_transform(my_df.drop(['name','artist','playcount','Unnamed: 0'],axis=1))
    reduction_df = pd.DataFrame(data=reduction_fit,columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])
    my_fit = pca.fit_transform(my_df.drop(['name','artist','playcount','Unnamed: 0'],axis=1))
    principalDf = pd.DataFrame(data = my_fit, columns = ['PC1','PC2'])
    principalDf = (principalDf.join(my_df['playcount'],how='left'))
    min_max = MinMaxScaler()
    print(pca.explained_variance_ratio_)
    #principalDf.ix[:, 'playcount'] = principalDf.ix[:, 'playcount'].apply(pd.to_numeric)
    #print(principalDf['playcount'].mean())
    #principalDf.loc['playcount'] = (principalDf.loc['playcount']-
    unscaled = principalDf[['playcount']].values
    scaled = min_max.fit_transform(unscaled)
    scaled_df = pd.DataFrame(scaled)
    principalDf['playcount'] = scaled_df
    #principalDf
    #my_df
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(2,1,1)
    plt.scatter(principalDf['PC1'],principalDf['PC2'],c=principalDf['playcount'],cmap='coolwarm')
    plt.colorbar()
    plt.show()

    tsne = TSNE(n_components=2,perplexity=30)
    tsne_fit = tsne.fit_transform(reduction_df)
    #tsne_fit = tsne.fit_transform(my_df.drop(['name','artist','playcount','Unnamed: 0'],axis=1))
    tsneDF = pd.DataFrame(data=tsne_fit, columns=['TSNE1','TSNE2'])
    tsneDF = tsneDF.join(my_df['playcount'],how='left')
    unscaled_tsne = tsneDF[['playcount']].values
    scaled_tsne = min_max.fit_transform(unscaled_tsne)
    tsneDF['playcount'] = scaled_tsne
    ax2 = fig.add_subplot(2,1,2)
    plt.scatter(tsneDF['TSNE1'],tsneDF['TSNE2'],c=tsneDF['playcount'],cmap='coolwarm')
    plt.colorbar()

top_2019 = "37i9dQZF1Etm3f2DGEBpQU"
littest_playlist="0yLY6eVDyGTyshTTkYaBKV"
def get_playlist_data(playlist):
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json',
                  'Authorization': 'Bearer BQCPvIHG-qckNcX1KTpm_qkJ4q4OX3QsXuNhnlnO5Gt6r-2URE_qRMcKxNyaKQIRXrBT-B-EzY-WlHdaH9aVq2_QUHRb4PcbNzdkAbfog12ADICovzLTTNVrpKVPAJdEpjQj0JBSRMSws90sNbjKYn41JgDC1B2tcFlI'}
    spotify_req = requests.get("https://api.spotify.com/v1/playlists/"+top_2019+"/tracks", headers=headers)
    offset = 0
    playlist = {}
    while True:
        spotify_url = "https://api.spotify.com/v1/playlists/"+littest_playlist+"/tracks?fields=items(added_at,track(id))&limit=100&offset="+str(offset)
        spotify_req = requests.get(spotify_url, headers=headers)
        if spotify_req.status_code != 200:
            print(spotify_req.status_code)
          # if song_tag_results.headers['Content-Length'] > 0
            try:
                print(spotify_req.json())
            except Exception as e:
                print(e)
            finally:
                print(spotify_url, spotify_req.content)
            continue
        if 'items' not in spotify_req.json():
            print(spotify_req.json())
            continue
        tracks = spotify_req.json()['items']
        if len(tracks) == 0:
            break
        ids = ''
        for i,track in enumerate(tracks):
            song_id = track['track']['id']
            if i < len(tracks)-1:
                ids+=song_id+','
            else:
                ids+=song_id
            playlist[song_id] = {'time':parser.parse(track['added_at']).timestamp()}
       # print(ids)
        audio_features_url = "https://api.spotify.com/v1/audio-features?ids="+ids
        audio_request = requests.get(audio_features_url,headers=headers)
        for track in audio_request.json()['audio_features']:
            playlist[track['id']] = {**playlist[track['id']], 'danceability': track['danceability'],
                                    'energy': track['energy'],
                                    'acousticness': track['acousticness'],
                                    'instrumentalness': track['instrumentalness'],
                                    'tempo': track['tempo'],
                                    'valence': track['valence'],
                                    'mode':track['mode'],
                                    'key':track['key']}

        offset+=100

def get_spotify_top_data():
    """
    Gets data from limited top tracks endpoint on spotify
    """
    top_tracks_url = "https://api.spotify.com/v1/me/top/tracks?limit=50&time_range=medium_term"
    playlist={}
    top_tracks_req = requests.get(top_tracks_url,headers=headers)
    top_ids=''
    #print(top_tracks_req.content,top_tracks_req.status_code)
    for i,track in enumerate(top_tracks_req.json()['items']):
        if i < len(top_tracks_req.json()['items'])-1:
            top_ids+=(track['id']+',')
        else:
            top_ids+=track['id']
    audio_features_url = "https://api.spotify.com/v1/audio-features?ids="+top_ids
    audio_request = requests.get(audio_features_url,headers=headers)
    print(audio_features_url)
    for track in audio_request.json()['audio_features']:
        playlist[track['id']] = {'danceability': track['danceability'],
                                    'energy': track['energy'],
                                    'acousticness': track['acousticness'],
                                    'instrumentalness': track['instrumentalness'],
                                    'tempo': track['tempo'],
                                    'valence': track['valence'],
                                    'mode':track['mode'],
                                    'key':track['key']}

def plot_spotify_data():
    playlist_df_unscaled = pd.DataFrame.from_dict(playlist,orient='index')
    #playlist_df_unscaled
    kPCA = KernelPCA(n_components=2)
    stand = StandardScaler()
    playlist_df = pd.DataFrame(stand.fit_transform(playlist_df_unscaled),index=playlist.keys(),
                               columns=['time','danceability','energy','acousticness',
                                       'instrumentalness','tempo','valence','mode','key'])
    #pca3 = PCA(n_components=3)
    #tsne3 = TSNE(n_components=3)
    #TSNE.set_params({'perplexity':20})
    pca_fit = pca.fit_transform(playlist_df.drop(['time'],axis=1))
    print(pca.components_[0],pca.components_[1])
    tsne_fit = tsne.fit_transform(playlist_df.drop(['time'],axis=1))

    print(tsne.fit(playlist_df))
    pca_df = pd.DataFrame(data = pca_fit, columns=['PC1','PC2'],index=playlist.keys())
    tsne_df = pd.DataFrame(data = tsne_fit, columns = ['TSNE1','TSNE2'], index=playlist.keys())
    analysis = pca_df.join(tsne_df,how='left')
    analysis = analysis.join(playlist_df, how='left')

    fig2 = plt.figure(figsize=(10,10))
    axP = fig2.add_subplot(311)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.scatter(analysis['PC1'],analysis['PC2'],c=analysis['energy'])
    plt.colorbar()
    #plt.show()
    #axL = fig2.add_subplot(122)
    #plt.scatter(pca.components_[0],pca.components_[1])
    axT = fig2.add_subplot(312)
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.scatter(analysis['TSNE1'],analysis['TSNE2'],c=analysis['energy'])
    plt.colorbar()

    axA = fig2.add_subplot(313)
    plt.xlabel('energy')
    plt.ylabel('valence')
    plt.scatter(playlist_df['valence'],playlist_df_unscaled['danceability'],c=playlist_df['mode'])
    plt.colorbar()

