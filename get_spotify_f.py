import os
import requests
import json
import access
import pandas as pd
import sys
from termcolor import colored
from itertools import chain


def init2():

	PATH = os.path.dirname(os.path.abspath(__file__))+'/'

	path_2_setlist = PATH+'pl_setlist/'
	headers = ['Song_ID','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']

	PL_data = PATH+'PL_DATA/'

	if not os.path.exists(PL_data):
		os.makedirs(PL_data)


	for user in os.listdir(path_2_setlist):
		
		##debugging	
		#if user == 'spotify':

			path_ = path_2_setlist+user+'/'
			for pl in os.listdir(path_):
				file = path_+pl
				print(" Path to Playlist Setlist File")
				print(file)
				print(" Playlist:")
				print(colored(pl[:-4],'cyan'))
				print("  by user:")
				print(colored(user,'cyan'))			
				
				## OUTPUT:
				output_file = PL_data+user+'/'+pl[:-4]+'/Spotify_features.tsv'

				###### 
				# if path not exist
				if not os.path.exists(PL_data+user+'/'+pl[:-4]+'/'):
					os.makedirs(PL_data+user+'/'+pl[:-4]+'/')

				# read playlist-setlist
				setlist = pd.read_csv(file,sep='\t')
				songs_list = setlist['Song2'].tolist()
				del songs_list[0]	## = ID 		
			
				## debugging:
				#print(len(songs_list))

				if len(songs_list) != 0:
					## check if Spotify Features already exist
					## update
					
					try:
						features_df = pd.read_csv(output_file,sep='\t')
						print(' Spotify Features prior')
						print(features_df)
						empty=False
						if len(features_df['Song_ID'])==0:# or len(features_df['Song_ID'])==1 : # 1->header? NO:see del songs_list[0] ## = ID 	
							empty = True
					except FileNotFoundError:
						### WRITE DataFrame
						print(colored(' File Not Found','red'))
						features_df = pd.DataFrame(columns = headers)
						empty=True
						######
					## debugging:
					print(len(songs_list))

					for song_id in songs_list:
						if song_id in features_df['Song_ID'].tolist():
								print(colored(song_id+' \n already in '+ output_file,'green'))
								songs_list.remove(song_id)
					
					for song_id in songs_list:	
						if song_id == 'not available':
							print(colored(song_id+' \n not available in Spotify ','magenta'))
							songs_list.remove(song_id)
				
					## debugging:
					print(len(songs_list))
					
					if len(songs_list) >= 100: 
						list_of_chunks = list(chunks(songs_list, 50))
						#print(len(list_of_chunks))
						#print(len(list_of_chunks[0]))

						## empty dataframe 
						features_ = []
						for chunks_ in list_of_chunks:
							print(colored(' Songs to extract Spotify Features: ','cyan'))
							print(chunks_)
							features_res = req_spotify_features(chunks_)
							#print(features_res)
							#print(type(features_))
							#sys.exit('here')						
							features_.append(features_res)    #,axis=0)#([features_,features_res],axis=0).drop_duplicates().reset_index(drop=True)
							
						## flatten list! 
						features__ = list(chain(*features_))
						#print(features_)

						#features_R = pd.DataFrame(columns = headers)
						for dict__ in features__:
							#features_R = pd.concat(df,axis=1,ignore_index=True)
							#print(dict__)
							
							append_to_df(output_file, dict__,empty)
					
					else: ## len(list) << 100
						print(colored(' Songs to extract Spotify Features (< 100 songs)','blue'))
						features_res = req_spotify_features(songs_list)
						for dict__ in features_res:
							append_to_df(output_file, dict__,empty)



#def req_spotify_songLIST_features(songs_list):
def append_to_df(output_file, features_res,empty):
	headers = ['Song_ID','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
	try:
		features_df = read_current_df(output_file)

	except FileNotFoundError:
		### WRITE DataFrame
		print(colored(' File Not Found','red'))
		features_df = pd.DataFrame(columns = headers)

	df__ = pd.DataFrame([features_res])#pd.DataFrame.from_dict(features_res, orient='columns')
	#print(df__)
	
	features_df = pd.concat([features_df,df__],axis=0).drop_duplicates().reset_index(drop=True)
	features_df.to_csv(output_file,sep='\t',index=False)


def read_current_df(output_file):
	features_df = pd.read_csv(output_file,sep='\t')
	return features_df

def req_spotify_features(song_id):
	headers_ = ['Song_ID','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']

	if type(song_id) == list:
		## A comma-separated list of the Spotify IDs for the tracks. Maximum: 100 IDs.
		## separated in packages
		string = '%2C'.join(song_id) ## %2C = ","
		
		url = "https://api.spotify.com/v1/audio-features/?ids="+string
		#print(url)

	else:
		url = "https://api.spotify.com/v1/audio-features/?ids="+song_id
	
	headers ={
    	"Authorization": "Bearer "+access_token,
    	}

	response = requests.request("GET", url, headers=headers)

	#########
	## debugging area:
	print(response)
	#print(response.text)
	#print(response.json())

	data = response.json()
	#print(data)
	#print(data['audio_features'])
	list_of_dics = data['audio_features']
	#print(list_of_dics)
	##########

	## empty dataframe 
	features_ = []

	for indx, song in enumerate(list_of_dics):
		#print(' - Song index')
		#print(indx)
		#print(song)
		if song == [None] or song == None:
			#print(response.text)
			try:
				features_dict = write_NaNs(song[0]['id'])
				features_.append(features_dict)			
			except TypeError:
				print(colored(" - TypeError: 'NoneType' object does not support item deletion # 1",'red'))
				pass
				#print(indx)
				#print(song)
				#print(list_of_dics)
		else:
			try:	
				features_dict = order_data(song)
				features_.append(features_dict)
			except TypeError:
				print(colored(" - TypeError: 'NoneType' object does not support item deletion # 2",'red'))
				#print(indx)
				print(song)
				#print(list_of_dics)

		

	#features_ = pd.concat(features_,axis=1,ignore_index=True) #features_res = pd.concat([features_,features_row],axis=0).drop_duplicates().reset_index(drop=True)

	return features_ 		## list of dictionaries. Each elem in list represents the features of the song. 


def write_NaNs(song_id):
	dict_ = {'Song_ID': song_id, 'acousticness': 'NaN', 'danceability': 'NaN', 'duration_ms':'NaN', 'energy': 'NaN', 'instrumentalness': 'NaN','key':'NaN', 'liveness': 'NaN','loudness':'NaN', 'mode':'NaN', 'speechiness': 'NaN', 'tempo':'NaN' ,'time_signature':'NaN', 'valence':'NaN' }
	return dict_
	#row_of_nans = pd.DataFrame([dict_])
	#print(row_of_nans)
	#return row_of_nans

# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
	# For item i in a range that is a length of l,
	for i in range(0, len(l), n):
		# Create an index range for l of n items:
		yield l[i:i+n]


def order_data(dict_):																																					## change this vvv 
	#print("... PRIOR:")																																					## VV POS: 1 VVVVV 
	headers = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','Song_ID']
	#print(dict_)
	#features_df = pd.DataFrame.from_dict(dict_)
	
	##change key = 'id' to 'Song_ID' + make it the first element
	del dict_['type']
	del dict_['uri']
	del dict_['analysis_url']
	del dict_['track_href']
	song_id = dict_.get('id')
	del dict_['id']

	d = dict_
	d['Song_ID'] =   song_id
	#print(d)
	return d

	#df = pd.DataFrame.from_dict([d])
	
	#return df



access_token = access.get_token()
init2()