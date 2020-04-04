import json

from termcolor import colored

import access

import requests

from datetime import datetime

import os

import csv

import time

import urllib.request, urllib.parse, urllib.error

import sys



preview_url_biglist 	=	 []
song_id_biglist 		=	 []

def init():
	### INPUT 																																
	user_list = ['spotify', 'spotifyusa', 'topsify','spotify_uk_','spotify_germany','spotify_espa%C3%B1a','spotify_france','kexp_official','hot97.1','chillhopmusic','bbc_playlister','npr_music','nme.com','mergerecordsofficial','subpoprecords','burgerrecords4life','warnerbros.records','matadorrecords']	
	#user_list = ['spotifycharts']
	### INPUT

	### date settings
	time = datetime.now()
	day = time.day
	month = time.month
	### date settings

	

	for user in user_list:

		### directory for the new date
		global path
		path = r'user_PL/'+user+'/'+str(month)+"_"+str(day)+"/"
		
		if not os.path.exists(path):
			os.makedirs(path)
		### directory for the new date
		
		access_token = access.get_token()

		user_pl(user, access_token)

	## ON PI DO NOT DOWNLOAD
	##download_2_DB(preview_url_biglist,song_id_biglist)


def user_pl(user,access_token):
	

	print(" ---------------------------------------------- ")
	print(colored(" "+user, 'cyan'))	
	#print(" "+access_token)
	print(" ---------------------------------------------- ")

	limit = 25


	while True:
		response = req_user(user, access_token,limit)

		if response.status_code == 200:

			print(colored(" Response Status Code: 200", 'green'))

			data = json.loads(response.text)

			items =  data['items']

			pl_list = get_pl(items)

			print(pl_list)
			print(colored(" "+user, 'cyan'))

			limit = limit+1

		

		if response.status_code != 200:
			print(colored(" ERROR: "+str(response.status_code), 'red'))
			
			data = json.loads(response.text)
			if 'Invalid limit' ==  data['error']['message']:
				print(' Reached Limit ')
				#print pl_list
				F = 0 
				while F < len(pl_list):
					pl_name = pl_list[F][1]
					
					if pl_name.startswith('This is') or pl_name.startswith('This Is'):
						print(colored(pl_name, 'magenta'))
						F = F+1				# saltar 
					else:
						pl_id 	= pl_list[F][0]
						main_req_playlist(user, pl_id, pl_name,access_token)
						F = F+1
				break

			else:
				print(data)
			 




def req_user(user, access_token,limit):
	

	url = 'https://api.spotify.com/v1/users/'+user+'/playlists?offset=0&limit='+str(limit)

	headers = {
		'Accept': "application/json",
		'Authorization': "Bearer "+access_token
	}

	response = requests.request('GET', url, headers=headers)

	return response


def get_pl(items):
	results = []
	
	for item in items:
		playlists 	=	item

		pl_name		=	playlists['name']

		pl_id		= 	playlists['id']

		if pl_name is None or pl_id is None:	# cath error
			print(colored(' Catch Error', 'red'))
					
		else:
			results.append(tuple((pl_id, pl_name)))

			

										# Results_old = [] to see if something changes
	return results	


def main_req_playlist(user, pl_id, pl_name, access_token):

	position_list 		= [pl_name, 'Pos.']
	### what is this for ? 
	#spotify_link 		= 'https://play.spotify.com/v1/users/'+user+'/playlists/'+pl_id
	### what is this for ? 
	song_list_name      = ['Song1','Name']
	song_list_id        = ['Song2','ID']
	artist_list_name    = ['Artist1','Name']
	artist_list_id      = ['Artist2','ID']
	album_list_name     = ['Album1','Name']			
	album_list_id       = ['Album2','ID']
	popularity_list     = ['Popularity']
	preview_url_list    = ['Preview_URL',user]

	print(" Next Playlist:")
	print(colored(" "+user, 'cyan'))
	print(colored(" "+pl_name, 'yellow')) 
	#print colored(" "+pl_id, 'red')
	

	response = req_playlist(user,pl_id,access_token)

	if response.status_code == 200:
		print(colored(" Response Status Code: 200", 'green')) 

		data = json.loads(response.text)
		
		### debuggging
		#print response.text
		###

		followers = data['followers']['total']
		popularity_list.append(followers)   

		items = data['tracks']['items']

		i = 0 
		while i < len(items):
			### debugging 
			#print items[i]
			###

			if items[i]['track']== None:
				print("fail")
				artists = 'not available'
				artists_name = 'not available'
				artists_id   = 'not available' 
				album        = 'not available'
				album_name   = 'not available'
				album_id     = 'not available'
				song_name    = 'not available'
				song_id      = 'not available'
				popularity   = 0
				position     = i+1
				preview_url  = 'not available'

			###
			else:
				added_at = items[i]['added_at']
				### this feature is not in used, but could be a nice addition. to other Spotify PL analyzers. 

				track = items[i]['track']
				
				### debugging 
				#print track['available_markets']
				###

				artists = track['artists']
				artists_name = artists[0]['name']
				artists_id   = artists[0]['id'] 
				album        = track['album']
				album_name   = album['name']
				album_id     = album['id'] 
				song_name    = track['name']
				song_id      = track['id']
				popularity   = track['popularity']
				position     = i+1
				preview_url  = track['preview_url']

			if preview_url is None:
				preview_url = "not available"

			print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

			print(' # '+ str(position))
			position_list.append(position)

			print(' Artist Name:            '+ colored(artists_name,'green')) 
			print(' Artist ID:              '+ colored(artists_id,'red'))

			artist_list_name.append(artists_name)
			artist_list_id.append(artists_id)

			print(' Album Name:             '+ colored(album_name,'green'))
			print(' Album ID:               '+ colored(album_id,'red'))

			album_list_id.append(album_id)
			album_list_name.append(album_name)

			print(' Song Name:              '+ colored(song_name,'green'))
			print(' Song ID:                '+ colored(song_id,'red'))

			song_list_id.append(song_id)
			song_list_name.append(song_name)
				
			print(' Preview:                '+ colored(preview_url,'cyan'))
			print(" ")
			preview_url_list.append(preview_url)
			print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

			

			#print '  '+ str(popularity)
			popularity_list.append(popularity)

				
			preview_url_biglist.append(preview_url)
			song_id_biglist.append(song_id)
			

			i = i + 1
		'''
		print(type(position_list))
		print(type(song_list_name))
		print(type(song_list_id))
		print(type(artist_list_name))
		print(type(artist_list_id))
		print(type(album_list_name))
		print(type(album_list_id))
		print(type(popularity_list))
		print(type(preview_url_list))
		print((position_list))
		print((song_list_name))
		print((song_list_id))
		print((artist_list_name))
		print((artist_list_id))
		print((album_list_name))
		print((album_list_id))
		print((popularity_list))
		print((preview_url_list))
		'''
		with open(path+pl_id+'.csv', 'w') as f:
			writer = csv.writer(f)
			writer.writerows(zip(position_list,song_list_name,song_list_id,artist_list_name,artist_list_id,album_list_name,album_list_id, popularity_list,preview_url_list))

	else:
		print(colored(" ERROR: "+ str(response.status_code),'red'))

		data = json.loads(response.text)
		print(data)




def req_playlist(user, pl_id, access_token):
	
	if '_' in user or '%' in user:			# 	  vvvvvvv 
		url_y = "https://api.spotify.com/v1/users/spotify/playlists/"+str(pl_id)

	else:
		url_y = "https://api.spotify.com/v1/users/"+user+"/playlists/"+str(pl_id)


	headers = {
		'Accept': "application/json",
		'Authorization': "Bearer "+access_token,
	}

	response = requests.request("GET", url_y, headers=headers)

	return response


def download_2_DB(preview_url_biglist,song_id_biglist):

	print(" ******************************** ")
	print(" DOWNLOAD TO DATABASE STARTING...")
	'''
	print " ******************************** "
	print preview_url_biglist
	print " ******************************** "
	print song_id_biglist
	print " ******************************** "
	'''
	
	pathmp3 = r'30s_DB/'

	if not os.path.exists(pathmp3):
		os.makedirs(pathmp3)

	already_downloaded_count = 0
	song_downloaded_count 	 = 0 
	not_available_count 	 = 0 


	start_time = datetime.now()

	for index, url in enumerate(preview_url_biglist): 

		if (url != "not available"):
	
			name_mp3 = pathmp3+str(song_id_biglist[index]) + ".mp3"

			print(song_id_biglist[index])
			

			if not os.path.exists(name_mp3):
				urllib.request.urlretrieve(url, name_mp3)
				print(colored("Song Downloaded",'green'))
				song_downloaded_count = song_downloaded_count + 1 
					
			
			else: 
				print(colored("Already Downloaded",'blue'))
				already_downloaded_count = already_downloaded_count + 1
		else: 
			not_available_count = not_available_count + 1 
					

	print(" -------------------------- ")
	time_spent = datetime.now() - start_time
	print(" TIME SPENT TO UPDATE DB 		: " + str(time_spent))
	print(" SONGS DOWNLOADED 				: " + str(song_downloaded_count))
	print(" SONGS ALREADY DOWNLOADED 		: " + str(already_downloaded_count))
	print(" SONGS NOT AVAILABLE 			: " + str(not_available_count))
	#print " NOT AVAILABLE PORCENTAGE 		: " + str(not_available_count/(already_downloaded_count+song_downloaded_count))


def debugger():
	access_token = access.get_token()
	main_req_playlist('spotify_france','37i9dQZF1DWU4xkXueiKGW','Fresh Rap',access_token)


#debugger()

def sleeper():
	sleep = True
	
	# Run our time.sleep() command,
	# and show the before and after time
	print('Before: %s' % time.ctime())
	print(colored('... waiting 24h','cyan'))
	time.sleep(86400)
	print('After: %s\n' % time.ctime())

	sleep = False


def waiter():
	while True:
		init()
		try:
			sleeper()
		except KeyboardInterrupt:
			print('\n\nKeyboard exception received. Exiting.')
			sys.exit()
#init()
#waiter()