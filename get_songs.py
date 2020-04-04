
from os import listdir
import os
from pathlib import Path
import pandas as pd
import csv
import sys
from termcolor import colored
import numpy as np



from operator import itemgetter

# global result directory of req_spotify.py
path_RES = r'user_PL/'

PATH = os.path.dirname(os.path.abspath(__file__))+'/'



## This part first checks for the most popular playlists of each user 
##  => popular_pl() --> get_followers() --> pl_to_inspect
## then a list of songs in each playlist is written --> to later extract the features of each song individually 
## 

def popular_pl():

	#print(path_RES)

	# 
	## DEVELOPING:
	## PRE get followeres
	for user in os.listdir(path_RES):
		if user.startswith('.'):
			pass
		else:
			print(path_RES+user)
			
			days = os.listdir(path_RES+user)
			print(days)
			try:
				pl_list = os.listdir(path_RES+user+"/"+days[1])
			except IndexError: 
				pl_list = os.listdir(path_RES+user+'/'+days[-1])

			for i, day in enumerate(days):
				if day.startswith("."):
					del days[i]

			for i, pl in enumerate(pl_list):
				if pl.startswith('.'):
					del pl_list[i]

			get_followers(user, days, pl_list)
	
	## PREPARE PLAYLISTS TO INSPECT by Popularity
	path_2_inspect = r'pl_to_inspect/'
	for user in listdir(path_2_inspect):
		if user.startswith('.'):
			pass
		else:
			print(" Path: "+r"pl_to_inspect/"+user)
			#df = pd.read_csv(r'pl_to_inspect/'+user)

			#print(df)
			with open(r"pl_to_inspect/"+user, 'r') as f:
				reader = csv.reader(f)
				pl_list = list(reader)
				pl_list = pl_list[0]

				for pl in pl_list:
					if len(pl)==0 or pl.startswith('.'):
						pass
					else:
						get_songs(user, pl)


def get_songs(user,pl):
	print(user[:-4])
	user = user[:-4]
	
	print(pl)
	#sys.exit()

	for day in listdir(path_RES+user):
		if not day.startswith('.'):
			#print(day)			
			#print(pl)
			#print(day)
			#print(user)
			#print(type(pl))
			#print(type(day))
			#print(type(user))
			#sys.exit()
			path_2_day = user+'/'+day+'/'+pl+'.csv'
							
			print(" ***** PLAYLIST = "+colored(pl,'green')+" *****")
			print(path_2_day)
			
			try:
				data = pd.read_csv(PATH+path_RES+path_2_day,header=0)
				df = data[['Song2','Artist1','Song1','Preview_URL']]

				print(df)
				pl_setlist(user,pl,df)

			except FileNotFoundError:
				print('Error: Could not find '+colored(path_2_day,'red'))

			

def pl_setlist(user,pl,df):

	path_2_playlist_setlist = r'pl_setlist/'
	path = path_2_playlist_setlist+user+'/'
	#print(os.path)
	if not os.path.exists(path):
		os.makedirs(path)

	file = Path(PATH+path+pl+'.tsv')
	#print(file)
	if not file.exists():
		df.to_csv(file,index=False,sep='\t')

	else:
		print(file)
		read = pd.read_csv(file,sep='\t')
		# dropping ALL duplicte values 
		
		
		result_df = pd.concat([read,df]).drop_duplicates().reset_index(drop=True)
		# droping duplicates

		result_df.to_csv(file,sep='\t',index=False)

	print(' -> done with setlist: '+ pl +' by '+user+' \n ')
	
	write_all_songs(user,pl,df)



def write_all_songs(user,pl,df):
	
	all_songs_file = Path(r'songs_data/all_songs.tsv')
	if not all_songs_file.exists():
		df.to_csv(all_songs_file,index=False,sep='\t')
	
	else:
		read = pd.read_csv(all_songs_file,sep='\t')

		result_df = pd.concat([read,df]).drop_duplicates().reset_index(drop=True)
		
		result_df.to_csv(all_songs_file,sep='\t',index=False)
		


def get_followers(user,days,pl_list):

	print(" **************************************** ")
	
	pl_follow_avrg = []
	
	for pl in pl_list:
		print(" PLAYLIST: "+colored(pl,'blue'))
		days_column = ['Days']
		popularity_column = ['Followers']
		pop = []

		sorting_list = []
		
		i = 0
		
		while i < len(days):
			path = path_RES+user+'/'+days[i]+'/'+pl
			print(" "+colored(path,'cyan'))
			try:
				df = pd.read_csv(path)
				popularity = int(df['Popularity'].iloc[0])
			except IOError:
				print(" FAIL FINDING PL:"+colored('ERROR--Code:1','red'))
				print(user)
				print(pl)
				fail_path = r"pl_popularity/deleted_pl.txt"
				with open(fail_path,'a') as fails:				## this should be emptied before the function get followers is called. BC it just appends the same stuff over and over
					fails.write(user+' , '+pl+' // ')
				break
			except ValueError:
				print(" FAIL FINDING PL:"+colored('ERROR--Code:2','red'))
				print(user)
				print(pl)
				fail_path = r"pl_popularity/deleted_pl.txt"
				with open(fail_path,'a') as fails:				## this should be emptied before the function get followers is called. BC it just appends the same stuff over and over
					fails.write(user+' , '+pl+' // ')
				break

			popularity_column.append(popularity)
			#print popularity_column
			#print popularity
			#print type(popularity)
			pop.append(popularity)
			days_column.append(days[i])

			month, day = days[i].split('_')


			sorting_list.append(tuple((day,month,popularity)))
			sorted_list = sorted(sorting_list, key=lambda element: (int(element[0]), int(element[1])))



			i = i+1

		#sys.exit(sorted_list)

		df = pd.DataFrame(sorted_list, columns=['Day','Month','Followers'])
		output_path = r'pl_popularity/'+user+'/'
		
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		
		df.to_csv(output_path+pl)


		
		avrg_pop = df['Followers'].mean()

		temp_tuple = (pl[:-4],avrg_pop)

		pl_follow_avrg.append(temp_tuple)

		
		## This is not finished:

		#print(days_column)
		#print(popularity_column)
		#print(type(days_column))
		#print(type(popularity_column))
		#sys.exit(output_path+pl)

		#with open(output_path+pl,'w') as f:
		#	df.to_csv(index=False)

		
	
	print(" *********************** ")
	print(pl_follow_avrg)
	print(" *********************** ")

	pl_to_inspect(user,pl_follow_avrg)
	

def pl_to_inspect(user,pl_follow_avrg):

	## reducing number of users:
	## change this to reduce the number of users to inspect! 
	if user == 'spotify':
		Num = 10
	else:
		Num = 5

	followers_l_sorted = sorted(pl_follow_avrg, key=lambda tup: tup[1])

	followers_tupl = followers_l_sorted[::-1]

	#print("PL of user "+user+" sorted:")
	#print(followers_tupl)
	#sys.exit()

	top = followers_tupl[:Num]		### GET TOP 15 PL
									### top is a variable of the top X pl with the most followers

	path_csv_file =  r"pl_to_inspect/"
	if not os.path.exists(path_csv_file):
			os.makedirs(path_csv_file)

	path_csv_file = open(path_csv_file+""+user+".csv",'w')

	for elemnt in top:
		path_csv_file.write("%s," %elemnt[0])



def init():

	popular_pl()
	
	#get_all_songs()   ## gathers all the data of the songs of the popular playlists! 


init()


