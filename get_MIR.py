import os
import pandas as pd
from pathlib import Path
import urllib
from termcolor import colored
import numpy as np
import collections
import sys

temp_path  = r'songs_data/temp/'
if not os.path.exists(temp_path):
	os.makedirs(temp_path)


## START MATLAB
print("	STARTING MATLAB ENGINE")
import matlab.engine
eng = matlab.engine.start_matlab()
#root_matlab = '/nas/ei/share/tools/matlab/R2019a'
eng.addpath('/home/ga59qek/MIRToolbox/')
print("		Engine running")


def download_30s(song_id,song_link):

	## TEMPORARY FILE 
	name_mp3 = temp_path+'TEMP'+'.mp3'
	
	## DOWNLOAD
	print('...downloading')
	try:
		urllib.request.urlretrieve(song_link, name_mp3)
	except OSError:
		print(" --> Retry")
		download_30s(song_id,song_link)

	print('Downloaded:')
	print("		"+colored(song_id+'.mp3','green'))


def init():

	PATH = os.path.dirname(os.path.abspath(__file__))+'/'

	path_2_setlist = PATH+'pl_setlist/'

	PL_data = PATH+'PL_DATA/'

	headers = ['Song_ID','spect_centroid_v' ,'rolloff85_mean', 'rolloff85_std','brightness_mean','brightness_std','spectralflux_mean','spectralflux_std','zerocross_rate','RMS_energy_mean','RMS_energy_std','lowenergy_v','tempo_v','pulseclarity_v','onsets_mean','onsets_std','onsets_peakpos','onsets_peakval','key_v','ASR_v', 'mfcc_mean' , 'mfcc_std','pitch_mean', 'pitch_std','event_dens_v', 'roughness_mean','roughness_std', 'cepstrum_mean','cepstrum_std','chroma_mean','chroma_std','key_st_mean','key_st_std','key_mode_v','ton_centroid_mean','ton_centroid_std']

	if len(set(headers))!=len(headers):
		print([item for item, count in collections.Counter(headers).items() if count > 1])
		sys.exit(print(colored('ERROR: duplicates in the headers','red')))

	if not os.path.exists(PL_data):
		os.makedirs(PL_data)

	for user in os.listdir(path_2_setlist):
		path_ = path_2_setlist+user+'/'
		for pl in os.listdir(path_):
			file = path_+pl
			print(file)

			###### 
			#PL_data+user+'/'+pl+'/'
			if not os.path.exists(PL_data+user+'/'+pl[:-4]+'/'):
				os.makedirs(PL_data+user+'/'+pl[:-4]+'/')		

			setlist = pd.read_csv(file,sep='\t')

			reduced_df = setlist[['Song2','Preview_URL']]
			print(reduced_df)
			
			for index, row in reduced_df.iterrows():
				song_id 	= row['Song2']
				song_link	= row['Preview_URL']
				if song_link.startswith('https://'):


					## check if Audio Features already exist
					## update
					output_file = PL_data+user+'/'+pl[:-4]+'/MIRaudio_features.tsv'
					try:
						audio_features_df = pd.read_csv(output_file,sep='\t')
						#print(' Audio Features prior')
						#print(audio_features_df)
						empty=False
						if len(audio_features_df['Song_ID'])==0:
							empty = True
					except FileNotFoundError:
						### WRITE DataFrame
						print(colored(' File Not Found','red'))
						audio_features_df = pd.DataFrame(columns = headers)
						empty=True
						######

					if song_id in audio_features_df['Song_ID'].tolist()	:
						print(colored('\n'+song_id+' \n already in '+ output_file,'green'))
					

					else:
						## download
						download_30s(song_id,song_link)
						##
						name_mp3 = temp_path+'TEMP'+'.mp3'
						##  MATLAB
						feat_vector = eng.features_ext_3(name_mp3)
						##
						#print(type(feat_vector))
						feat_ = list(np.array(feat_vector._data))
						#print(feat_)
						feat_.insert(0, song_id)
						
						print(' -- Audio Features --')
						#print(feat_)
						#print(len(feat_))
						#print('LEN Headers: '+str(len(headers)))
						print(' --     (as DF)    --')

						df = pd.DataFrame([feat_],columns=headers)
						
						print(df)
						print(' --                --')
						print(audio_features_df)
						#print('compare')
						#sys.exit()
						#frame = [audio_features_df,df]
						#result_df = pd.concat(frame,sort=False).reset_index(drop=True)

						#print(result_df)
						if empty== True:
							audio_features_df = df
						else:
							audio_features_df = pd.concat([audio_features_df,df],axis=0).drop_duplicates().reset_index(drop=True)
 

					

						audio_features_df.to_csv(output_file,sep='\t',index=False)



init()




'''
def init_old():

	PATH = os.path.dirname(os.path.abspath(__file__))+'/'

	path_2_allsongs = PATH+'songs_data/all_songs.tsv'
	file = Path(PATH+'songs_data/songs_MIRfeatures.tsv')

	### WRITE DataFrame
	headers = ['Song_ID','spect_centroid_v' ,'rolloff85_mean', 'rolloff85_std','brightness_mean','brightness_std','spectralflux_mean','spectralflux_std','zerocross_rate','RMS_energy_mean','RMS_energy_std','lowenergy_v','tempo_v','pulseclarity_v','onsets_mean','onsets_std','onsets_peakpos','onsets_peakval','key_v' ,'key_mode_v' , 'ASR_v', 'mfcc_mean' ,'mfcc_std' ,'pitch_mean', 'pitch_std','event_dens_v', 'roughness_mean','roughness_std','cepstrum_mean','cepstrum_std','chroma_mean','chroma_std', 'key_st_mean','key_st_std','key_mode_v','ton_centroid_mean','ton_centroid_std']
	
	## read TO DO !!! 

	audio_features_df = pd.DataFrame(columns = headers)

	if not file.exists():
		audio_features_df.to_csv(file,index=False,sep='\t')

	else:
		#print(file)
		audio_features_df = pd.read_csv(file,sep='\t')
	

	all_songs = pd.read_csv(path_2_allsongs,sep='\t')

	reduced_df = all_songs[['Song2','Preview_URL']]
	print(reduced_df)

	for index, row in reduced_df.iterrows():
		song_id 	= row['Song2']
		song_link	= row['Preview_URL']
		if song_link.startswith('https://'):

			## update
			audio_features_df = pd.read_csv(file,sep='\t')

			## download
			download_30s(song_id,song_link)
			##
			name_mp3 = temp_path+song_id+'.mp3'
			##  MATLAB
			feat_vector = eng.features_ext_3(name_mp3)
			##
			#print(type(feat_vector))
			feat_ = list(np.array(feat_vector._data))
			#print(feat_)
			feat_.insert(0, song_id)
			
			print(' -- Audio Features --')
			print(feat_)
			print(' --     (as DF)    --')

			df = pd.DataFrame([feat_],columns= headers)
			
			print(df)
			print(' --                --')

			result_df = pd.concat([audio_features_df,df])
			print(result_df)

			

			result_df.to_csv(file,sep='\t',index=False)
'''
