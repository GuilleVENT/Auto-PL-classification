import os
import pandas as pd
import re
import sys

from termcolor import colored

## language detection
from langdetect import detect
import collections
from itertools import dropwhile
from itertools import chain


## NLTK for text processing
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download('stopwords')

## Wordcloud:
#from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

## my own dictionaries
import get_vocabulary



PATH = os.path.dirname(os.path.abspath(__file__))+'/'


def get_lyrics_from_txt(file):

	with open(file) as f:
		content = f.readlines()

	lyrics = [x.strip() for x in content] ## each line is an element in a list

	structure = []

	for line in lyrics:
		if line.startswith('['):
			structure.append(line)
			lyrics.remove(line)
		elif line.startswith('[?]'):
			pass
		'''
		if len(line)==0:
			lyrics.remove(line)
		'''

	#print(" 	Song Structure: 	  	")
	#print(structure)

	return lyrics, structure 			### lyrics and *structure* !


def get_estrofas(lyrics,structure):
	#print(lyrics)

	total_num_estrofas = len(structure)  # == lyrics.count('')-1
	
	## cleaning...
	i=0
	while i < len(lyrics):
		try:
			if lyrics[i] == '' and lyrics[i+1] == '' and lyrics[i]==lyrics[i+1]:
				del lyrics[i]
			i+=1
		except IndexError:
			i+=1
	
	## count "" strategy. That's why doble (or tripple) "spaces-lines" needed to be cleaned

	verses_per_estrofa = []
	counter = 0
	indx = 0 ## i think this will solve the bug of: SONG:   Space Cadet 			ARTIST: Philanthrope

	for indx,verse in enumerate(lyrics):
		#print(indx)
		#print(verse)
		
		if verse == '':
			counter = 0 # reset
			#print('reset')
		else:
			counter += 1
			#print('+')
		try:
			if lyrics[indx+1] == '':
				verses_per_estrofa.append(counter)
		except IndexError:
			pass
		#idx+=1

	#print(verses_per_estrofa)
	if verses_per_estrofa[:-1]==0:
		verses_per_estrofa.pop()

	if len(verses_per_estrofa)>15:
		verses_per_estrofa=verses_per_estrofa[:15]
	
	try:
		min_verses_per_estrofa = min(verses_per_estrofa)
		max_verses_per_estrofa = max(verses_per_estrofa)
		avg_verses_per_estrofa = float(sum(verses_per_estrofa)/len(verses_per_estrofa))
	except:
		min_verses_per_estrofa = 0
		max_verses_per_estrofa = 0
		avg_verses_per_estrofa = 0

	print(' - Verses Per Estrofa:')
	print(verses_per_estrofa)
	
	## for DF formatting reasons:
	while len(verses_per_estrofa) < 15:
		verses_per_estrofa.append(0)
		indx += 1

	


	print(' - Shortest Estrofa length:')
	print(min_verses_per_estrofa)
	print(' - Longest Estrofa length:')
	print(max_verses_per_estrofa)
	print(' - Average number of Verses per Estrofa')
	print(avg_verses_per_estrofa)




	# not returning this "verses_per_estrofa" ->type = list
	#return total_num_estrofas, max_verses_per_estrofa, min_verses_per_estrofa, avg_verses_per_estrofa, verses_per_estrofa
	'''
	df_res0 = pd.DataFrame([verses_per_estrofa],columns=['E01','E02','E03','E04','E05','E06','E07','E08','E09','E10','E11','E12','E13','E14','E15'])
	#print(df_res0)
	df_res1 = pd.DataFrame(data=[[total_num_estrofas, max_verses_per_estrofa, min_verses_per_estrofa, avg_verses_per_estrofa]],columns=['total_num_estrofas','max_verses_per_estrofa','min_verses_per_estrofa','avg_verses_per_estrofa'])
	#print(df_res1)

	#df_res = df_res1.append([df_res0])
	df_res = pd.concat([df_res0,df_res1.reindex(df_res0.index)], axis=1)
	'''
	return verses_per_estrofa, min_verses_per_estrofa, max_verses_per_estrofa, avg_verses_per_estrofa, total_num_estrofas
	
def structure_features(lyrics,structure):

	parts = []
	indx = 0
	if len(structure) != 0:
		for indx, part in enumerate(structure):
			if indx < 15:
				if 'intro' in part.lower():
					print(1.0)
					parts.append(1.0)
				elif 'verse' in part.lower():
					int_ = detect_number(part.lower())
					print(2.0+int_/10)
					parts.append(2.0+int_/10)
				elif 'chorus' in part.lower():
					if 'pre' in part.lower():
						print(3.1)
						parts.append(3.1)
					else:
						print(3.0)
						parts.append(3.0)
				elif 'bridge' in part.lower():
					print(4.0)
					parts.append(4.0)
				elif 'refrain' in part.lower():
					if 'pre' in part.lower():
						print(5.1)
						parts.append(5.1)
					else:
						print(5.0)
						parts.append(5.0)
				elif 'couplet' in part.lower():
					print(6.0)
					parts.append(6.0)
				elif 'interlude' in part.lower():
					print(7.0)
					parts.append(7.0)
				elif 'drop' in part.lower():
					if 'pre' in part.lower():
						print(8.1)
						parts.append(8.1)
					else:
						print(8.0)
						parts.append(8.0)
				elif 'hook' in part.lower():
					if 'pre' in part.lower():
						print(9.1)
						parts.append(9.1)
					else:
						print(9.0)
						parts.append(9.0)
				elif 'break' in part.lower():
					print(10.0)
					parts.append(10.0)
				elif 'solo' in part.lower():
					if 'guitar' in part.lower():
						print(11.1)
						X = 11.1
					if 'bass' in part.lower():
						print(11.2)
						X = 11.2
					if 'key' or 'piano' in part.lower():
						print(11.3)
						X = 11.3
					if 'drum' in part.lower():
						print(11.4)
						X = 11.4
					else:
						print(11.5)
						X = 11.5
					parts.append(X)
				elif 'instr' in part.lower():
					print(11.0)
					parts.append(11.0)
				elif 'outro' in part.lower():
					print(12.0) 
					parts.append(12.0)
				else:
					print(13.0) # = other
					parts.append(13.0) ## other mala suerte 
					
		while indx < 14:
			parts.append(0.0)
			indx += 1

		duet_ = []
		for part in structure:
			if ':' in part:
				splitted = part.split(':',1)
				
				part   = splitted[0]
				artists = splitted[1]

				if ',' in artists or '&' in artists:
					duet = True
					duet_.append(duet)
				else:
					duet = False	
					duet_.append(duet)
			else:
				artists = '-'
				duet = False
				duet_.append(duet)

		if any(x == True for x in duet_):
			DUET = 1
		else:
			DUET = 0
		
		
		#print(parts)

		#print(DUET)

		#print(parts)
		'''
		df_res = pd.DataFrame([parts],columns=['P01','P02','P03','P04','P05','P06','P07','P08','P09','P10','P11','P12','P13','P14','P15','Duet'])
		print(df_res)
		'''
		return parts,DUET

	else:
		print('	No structure available... writing zeros')
		return([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],0)



def word_cloud(file_name,lyrics): ## 
	if type(lyrics)==list:
		lyrics = ', '.join(list(filter(None, lyrics)))

	elif os.path.isfile(lyrics):
		lyrics, structure = get_lyrics_from_txt(lyrics_file)
		lyrics = ', '.join(list(filter(None, lyrics)))
	else:
		print('TYPE')
		print(type(lyrics))
		sys.exit('--> exit')
	
	#print(lyrics)

	if not os.path.exists(PATH+'WordClouds/'):
		os.makedirs(PATH+'WordClouds/')

	file_name = PATH+'WordClouds/'+file_name
	wordcloud = WordCloud(collocations=False,width=1200, height=1000, stopwords=STOPWORDS,).generate(lyrics)

	wordcloud.to_file(file_name)

'''
lyrics_file = PATH+'lyrics/'+'Prince'+'/'+'Purple Rain'+'.txt'
lyrics, structure = get_lyrics_from_txt(lyrics_file)
lyrics = ', '.join(list(filter(None, lyrics)))
print(lyrics)
#sys.exit('here')
word_cloud('purple_rain.png',lyrics)
'''

def get_words(lyrics_,Lang):
	#print('.........')
	lyrics = list(chain(*lyrics_)) ## unnest 

	print(lyrics)

	sum_ = len(lyrics)
	print('- Total Number of words:')
	print(sum_)
	if sum_ == 0:
		return(sum_,0,0,0,0)

	'''if Lang != 'en':
		pass
		#return(sum_,float('NaN'),float('NaN'),float('NaN'),float('NaN'))
		
	
	else: ## lang == 'en'''
	## "BAG OF WORDS" APPROACH
	pop_words = get_vocabulary.of_pop()
	pop_count = 0
	metal_words = get_vocabulary.of_metal()
	metal_count = 0
	happy_words = get_vocabulary.happy()
	happy_count = 0 
	hiphop_words = get_vocabulary.hiphop_slang()
	hiphop_count = 0
	
	##	TO DO :
	##	Sad words dictionary
	##

	for i, word in enumerate(lyrics):
		#print(i)
		#print(word)
		if word in pop_words:
			#print('POP')
			#print(word)
			pop_count += 1
		if word in metal_words:
			#print('METAL')
			#print(word)
			metal_count += 1 
		if word in happy_words:
			#print('HAPPY')
			#print(word)
			happy_count += 1
		if word in hiphop_words:
			#print('HIPHOP')
			#print(word)
			hiphop_count += 1

	pop_100 = float(pop_count / sum_)
	metal_100 = float(metal_count / sum_)
	happy_100 = float(happy_count / sum_)
	hiphop_100 = float(hiphop_count / sum_)
	
	print(' - - - - - - word class - - - - - - ')
	print('POP  % :\n'+str(pop_100))
	print('METAL %  :\n'+str(metal_100))
	print('HAPPY % : \n'+ str(happy_100))
	print('HIPHOP % : \n'+str(hiphop_100))
	print(' - - - - - - word class - - - - - - ')

	#df_res = pd.DataFrame([[sum_,pop_100,metal_100,happy_100,hiphop_100]],columns=['Total_Words','Pop_words_100','Metal_words_100','happy_words_100','Hiphop_words_100'])

	return sum_,pop_100,metal_100,happy_100,hiphop_100
	

def get_lengths(lyrics):			
	#
	lyrics_tokens = list(filter(None, lyrics))

	try:

		## shortest 
		shortest_verse 		= min(lyrics_tokens, key=len)
		shortest_verse_len 	= len(shortest_verse)

		print(' Shortest Verse Length:')
	
	

		print(shortest_verse_len)
		
		## longest

		longest_verse  		= max(lyrics_tokens, key=len)
		longest_verse_len 	= len(longest_verse)

		print(' Longest Verse Length:')

	
		print(longest_verse_len)
		verse_length = []
		for verse in lyrics_tokens:
			verse_length.append(len(verse))
		#print(verse_length)
		avrg_verse_length = float(sum(verse_length)/len(verse_length))
		print(' Avrg Verse Length:')
		print(avrg_verse_length)

	except:		## this is the case for instrumental songs 
		shortest_verse_len = 0
		longest_verse_len = 0
		avrg_verse_length = 0  

	#df_res = pd.DataFrame([[shortest_verse_len, longest_verse_len, avrg_verse_length]],columns=['shortest_verse_len', 'longest_verse_len', 'avrg_verse_length'])
	return shortest_verse_len, longest_verse_len, avrg_verse_length

def get_repetitions(song):
	## unnesting lyrics:
	count = 0
	seen = []
	for verse in song:		## get the amount of repeated verses!
		if len(verse)==0:
			pass
		else:
			if verse in seen:
				count=count+1
			else:
				#print(" - Verse:")
				#print(verse)
				seen.append(verse)

	#print(seen)
	#print(len(seen))
	#print(len(song))
	Num_repeated_verses = count
	print(" - Number of identical verses:")
	print(Num_repeated_verses)

	# reset seen:
	seen = []
	counter = 0
	for verse in song:
		if len(verse) == 0:
			pass 
		else:
			for word in verse.split(' '):
				counter += 1
				if word not in seen:
					if word.endswith(',') or word.endswith('.') or word.endswith(';') or word.endswith('?') or word.endswith('!'):
						word[:-1]
					#print(word)
					seen.append(word)
	
	
	if counter == 0:
		repe_100 = 0
	else:
		repe_100 = float(len(seen)/counter)*100

	print(" - Porcentage of repeated words:")
	print(repe_100)
	
	#return(pd.DataFrame([[repe_100, Num_repeated_verses]],columns=['Repetition_100','Repeated Verses']))
	return repe_100, Num_repeated_verses

def text_preprocessing(lyrics):		## lyrics == type: LIST OF VERSES! 

	stemmer = WordNetLemmatizer()
	new_lyrics = []

	for line in lyrics:

		# Remove all the special characters
		new_line = re.sub(r'\W', ' ', str(line))
		

		# remove all single characters
		new_line = re.sub(r'\s+[a-zA-Z]\s+', ' ', str(new_line))

		# Substituting multiple spaces with single space
		new_line = re.sub(r'\s+', ' ', new_line, flags=re.I)

		# Removing prefixed 'b'
		new_line = re.sub(r'^b\s+', '', new_line)

		# Converting to Lowercase
		new_line = new_line.lower()
		
		# Lemmatization
		new_line = new_line.split()  	# we reduce the word into dictionary root form. For instance "cats" is converted into "cat". Lemmatization is done in order to avoid creating features that are semantically similar but syntactically different. For instance, we don't want two different features named "cats" and "cat", which are semantically similar, therefore we perform lemmatization.

		document = [stemmer.lemmatize(word) for word in new_line]
		document = ' '.join(document)
		
		new_lyrics.append(new_line)

	
	print(new_lyrics)

	return new_lyrics


def get_stopWords(lyrics_tokens,language):

	## NLTK StopWords available in 
	'''
	['hungarian',
	 'swedish',
	 'kazakh',
	 'norwegian',
	 'finnish',
	 'arabic',
	 'indonesian',
	 'portuguese',
	 'turkish',
	 'azerbaijani',
	 'slovene',
	 'spanish',
	 'danish',
	 'nepali',
	 'romanian',
	 'greek',
	 'dutch',
	 'tajik',
	 'german',
	 'english',
	 'russian',
	 'french',
	 'italian']
 	'''
	if language == 'en':
		stopWords = set(stopwords.words('english'))
	elif language == 'es':
		stopWords = set(stopwords.words('spanish'))
	elif language == 'de':
		stopWords = set(stopwords.words('german'))
	elif language == 'du':
		stopWords = set(stopwords.words('dutch'))
	elif language == 'ru':
		stopWords = set(stopwords.words('russian'))
	elif language == 'it':
		stopWords = set(stopwords.words('italian'))
	elif language == 'fr':
		stopWords = set(stopwords.words('french'))
	elif language == 'no':
		stopWords = set(stopwords.words('norwegian'))
	#elif language == 'sw':
	#	stopWords = set(stopwords.words('sweedish'))
	elif language == 'ar':
		stopWords = set(stopwords.words('arabic'))
	else:
		return(0)

	#print(lyrics_tokens)
	# unnesting verses and estrofas
	lyrics_tokens = sum(lyrics_tokens,[])
	#print(lyrics_tokens)

	stopwords_x = []

	for word in lyrics_tokens:
		if word in stopWords:
			stopwords_x.append(word)

	stopWords_100 =len(stopwords_x) / len(lyrics_tokens) * 100
	
	print(" - % of stopwords ")
	print(stopWords_100) ## porcentage of stop words
	
	return(stopWords_100)


def get_language(lyrics): ## lyrics == type: LIST OF VERSES! 
	try:
		#print(lyrics)
		language = []

		for verse in lyrics:
			if len(verse)==0:
				pass
			elif 'instrumental' in verse.lower():
				pass
			else:
				try:
					lang = detect(verse)
					language.append(lang)
				except:
					lang = 'ff'
					language.append(lang)
		#print(language)
		dict_ = collections.Counter(language)
		print(dict_)
		language = dict_.most_common(1)[0][0]

		language_encoded = numerical_language(language)



		## this is to get if there are two languages used in the songs
		lang_top2 = dict_.most_common(2)
		print(lang_top2)
		if len(lang_top2)==2:
			lan1 = lang_top2[0][0] 
			lan2 = lang_top2[1][0]
			if lan1 and lan2 in ['es','en','de','fr','sw','ru','ar','du','no','it']:
				## amount of verses the second most used language is used:
				if lang_top2[1][1] > len(lyrics)*0.05: ## thershold: must surpase 5% of the song length 
					lang_mix = 1
				else:
					lang_mix = 0
			else:
				lang_mix = 0
		else:
			lang_mix = 0

		return language_encoded,lang_mix,language		## numerical encoding
									## lang_mix is bool. it determines if a song contains more than one language
	except:
		return(0,0,0)

def numerical_language(lang):
	if lang == 'en':
		return 1
	if lang == 'de':
		return 2
	if lang == 'es':
		return 3
	if lang == 'du':
		return 4
	if lang == 'ru':
		return 5
	if lang == 'it':
		return 6
	if lang == 'fr':
		return 7
	if lang == 'no':
		return 8
	if lang == 'ar':
		return 9
	if lang == 'sw':
		return 10
	else:
		return 0


def detect_number(verse):
	try:
		i_ = int(re.search(r'\d+', verse).group())
	except AttributeError:
		i_ = 0.1

	return i_

## developing:
# song hardcoded -	

#path2song = PATH+'lyrics/Led Zeppelin/Black Dog - Remaster.txt'
#with open(path2song) as f:
#	content = f.readlines()

#tontofile= '/Users/guillermoventuramartinezAIR/Desktop/FP/lyrics/Lil Dicky/Earth.txt'
#l,s=get_lyrics_from_txt(tontofile)
#get_estrofas(l,s)
#structure_features(l,s)







'''
def init():

	

	path_2_setlist = PATH+'pl_setlist/'
	PL_data = PATH+'PL_DATA/'

	for user in os.listdir(path_2_setlist):
		
		## developing:
		## user hardcoded	
		if user == 'spotify':

			path_ = path_2_setlist+user+'/'
			for pl in os.listdir(path_):
				file = path_+pl
				print(" Path to Playlist Setlist File")
				print(file)
				print(" Playlist:")
				print(colored(pl[:-4],'cyan'))
				print("  by user:")
				print(colored(user,'cyan'))

				###### 
				# if path not exist
				if not os.path.exists(PL_data+user+'/'+pl[:-4]+'/'):
					os.makedirs(PL_data+user+'/'+pl[:-4]+'/')

				# read playlist-setlist
				setlist = pd.read_csv(file,sep='\t')
				print(setlist)
				#sys.exit()
				
				songs_list = setlist['Song1'].tolist()

				del songs_list[0]	## = ID

				artists_list = setlist['Artist1'].tolist()
				del artists_list[0]

				print(songs_list)
				print(artists_list)

				
				for index, song in enumerate(songs_list):
					change_names(song)

				## from get_lyrics
def change_names(song_name)

		if '(' in song_name:
			song_title_w_feat = song_name 
			print(colored('song with parenthesis','red'))
			song_name_ = re.sub(r'\([^)]*\)', '', song_name) ## featuring song-> artist name in parenthesis.
			song_name = song_name_
			print(' Tweaked song name:')
			print(song_name)

		if '-' in song_name: ## = radio edits // remixes - let's get OG
		
			song_name_ = song_name.split('-',1)
			song_name = song_name_[0]
			print(' Tweaked song name:')
			print(song_name)

def structure_features_OLD(lyrics,structure):
	
	#print(structure)
	print(' - - - - - - - ')
	file = PATH+"structural_analysis.tsv"
	duet_ = []
	
	if os.path.isfile(file):
		struct_df = pd.read_csv(file)
	else:
		struct_df = pd.DataFrame(columns = ['Structure','Artist','Duet'])#empty
	
	for part in structure:
		
		#print(part)
		part = part[1:][:-1]

		if ':' in part:
			splitted = part.split(':',1)
			
			part   = splitted[0]
			artists = splitted[1]

			if ',' in artists or '&' in artists:
				duet = True
				duet_.append(duet)
			else:
				duet = False	
				duet_.append(duet)
		else:
			artists = '-'
			duet = False
			duet_.append(duet)

		if any(str(num) in part for num in list(range(1,10))): 	## number of verse [or other]
				#print(part)
				part_ = part[:-2] 	## this only works on nums < 9 (einstellig)
				part = part_
				#print(part)

		di = {'Duet':duet,'Artist':artists,'Structure':part}
		d = pd.DataFrame([di])
		#print(d)
		if part not in struct_df['Structure'].tolist():
			struct_df = pd.concat([struct_df,d]).drop_duplicates().reset_index(drop=True)

	
	## file to be written (pretty)
	struct_df.drop(struct_df.columns[struct_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
	#print(struct_df)
	struct_df.to_csv(file)



'''


# init()
#get_lyrics_n_structure()