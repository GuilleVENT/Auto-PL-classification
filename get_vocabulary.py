import nltk 
from nltk.corpus import wordnet 
import sys
import csv

## would be cool to include this:
## https://github.com/tmthyjames/cypher


def synonyms(in_word):
	synonyms = []

	for syn in wordnet.synsets(in_word): 
		for l in syn.lemmas(): 
			synonyms.append(l.name())

	return set(synonyms)



def of_pop():
	pop_words = []
	fileName = r'dictionary/pop-words.txt'
	
	for line in open(fileName):
		if line.startswith('#'):
			pass
		else:
			pop_words.append(line[:-1])
	
	return pop_words

def of_metal():
	metal_words = []
	fileName = r'dictionary/metal-words.txt'
	
	for line in open(fileName):
		if line.startswith('#'):
			pass
		else:
			metal_words.append(line[:-1])
	
	return metal_words

def happy():

	happy_words = []
	fileName = r'dictionary/happy-words.txt'
	
	for line in open(fileName):
		if line.startswith('#'):
			pass
		else:
			happy_words.append(line[:-1])
	
	return happy_words




def hiphop_slang():

	## taken from here:
	## https://www.dreadpen.com/hip-hop-slang-dictionary/

	hiphop_slang = [
	'wifey', #– girlfriend
	'kicks',# – sneakers
	'Big Face',#– 100 Dollar Bill
	'baller'# -n.- a high-roller, a money-maker
	'chillax'#-v.- relaxing. A combination of “chill” and “relax.
	'diss'#-v.- to criticize or disrespect someone. (e.g. “He dissed that bitch.” )
	'hella'#-adv.- very formed from “hell” and “of”; not used much by M.C.s anymore; common on theWest Coast
	"Mo’"," Fo’",'motherfucker',#-n.- motherfucker
	'scrilla',#-n.- money; used especially by West Coast M.C.s. (Mack 10, E-40, etc.)
	#Hip hop slang for drugs, gangs and guns
	#
	#Gang slang terms
	#
	'BG', #-n.- Baby Gangsta; an adolescent gangster
	'bluh',#-n.- a slurred pronunciation of Blood. Generally means friend, homie, fellow Bloods member. Usually used in phrases such as “What up, bluh?”, meaning “What’s up, Blood?” Generally used to refer to a Blood gang member, but sometimes used by Bloods towards non-Bloods gang members to provoke confrontation.
	'Cuzz','Cuzzo', #-n.- Crip. Sometimes pronounced “cuh”. A familiar term between members of the Crip gang, it can also be used in a confrontational manner from a Crip gang member towards a non-Crip.
	'G',# -n.- a gangsta. (I’m a G, I’m a G) in East Coast or “old school” hip-hop can mean simply a guy or girl.
	'O',#-n.- Original Gangster. Initially referred to the founder of a street gang, but now commonly refers to any older gang member. Usually common to gang bangers who live up to their notoriety, reputation, and never “sold out”. Has been adopted outside of the gang culture for general use in hip-hop to refer to any originator of something or older person.
	'overho',# -n.- derogatory term towards a Ova Soldier gang member.
	'suwitchbo',# -n.- derogatory term towards a D.T.B ganster.
	'Dru',#-related slang terms
	#
	'Papi',# – The drug connect
	'White Lady',# – Cocaine
	'Dope boy',# – Drug Dealer
	'boi',# – heroin
	'cookies', #– crack cocaine
	'nick',# (also “nickel”, “nickelbag”, “nickelsack”) – a five dollar bag of illicit drug
	'dime',# (also “dimebag” or “dimesack”) – ten dollar bag of illicit drugs
	'fire', 'marijuana','weed', 'marihuana',# or meaning a sex term towards oral sex
	'green',#– marijuana
	'primo',# – a joint laced with angel dust or crack cocaine
	'powdering', #– snorting cocaine
	'snort',#– snorting cocaine
	'Gun',#-related slang terms
	#
	'Glock', 'handgun',# – nine, nina, Nina Ross
	'.22', 'caliber', 'gun' #– Deuce Deuce, Scooby-Doo (used by Cypress Hill)
	'40-caliber','pounda',# gun #– 4 pounda
	'44-caliber','44',# gun #– 44, Fo’ Fo’
	'45-caliber',# gun #– Fo’Five
	'Desert', 'Eagle',#- Desert Eaze, Deagle
	'shotgun',#- pump, shotty, “The Dimple-ator”
	#Hip hop slang for brand names and trademarks
	#
	'Benz','Benzo',#– short for Mercedes-Benz
	'Beamer',#– any model of BMW vehicle
	'Cad','Caddy','’Lac', #– a Cadillac
	'Deuce',# and a Quarter – a Buick Electra
	'Dom P','Dom Perignon',#.# -Dom Perignon, a brand of champagne
	'Henny','Hennessy',#, Hen – Hennessy, a brand of cognac
	'L dog', #– a Lincoln
	'Lex',# – short for Lexus, also short for Rolex watches
	'Timbs',#– Timberland boots
	'Rolly','Rollex',# (also Rolley) – a term for Rolex, as used in Snoop Dogg’s 2004 hit “Drop It Like It’s Hot”. Also a synonym for Rolls Royce.
	#
	#Meaning of numbers #in hip hop slang
	#
	'187',#- homicide
	'24/7', #– all day hustle
	'411',# – information. From 4-1-1, the number for directory assistance in the United States.
	'420',#– Number associated with cannabis and cannabis usage. Also refers to April 20th, which many people refer to as “Marijuana Day” because of the date.
	'5-0',# – in reference to the police (as in the television show Hawaii Five-O)
	'1096'#– police code for mentally ill suspect (from the film The Sugarland Express). Commonly used in the state of Texas.
	'150', #– code for mentally ill from the California Welfare and Institutions Code. Also the name of a Van Halen and an Eazy-E album.
	'730',# – the code for a crazy person. According to some New York City rappers the term “730” originated from mental health patients receiving their medication at 7:30 am and 7:30 pm.
	'211',# – robbery
	'20',# – location (e.g. “What’s yo’ 20?” = “Where are you?”)
	#
	## 
	## taken from: https://www.the-art-of-small-talk.com/hiphopslang.html
	# pronunciation dependent
	#
	'aight','aite','aw-ite',#- all right.
	'axe',#-ask "I want to axe you a question."
	'befo',#-before "I told you this befo'."
	'da',#-the "He's da' man."
	'dat',#-that "I love dat."
	'dis',#-this "I love dis."
	'down',#I'm up for it. "I'm down."
	'flava',#-flavor.
	'flow',#-to rap,can also be used as a noun,"Listen to myflow."
	'flo',#'-floor.
	'fo',#(1)-for "This is fo' my Aunt."
	'fo',#(2)-the number four.
	'fo',#(3)-short for before,or any fore suffix.
	'mo',#-more "I can't stand it no mo."
	'moo',#-move "moo’ over there for me."
	'sup','wadup',

	'chillax',#-'chilling' while relaxing.
	'white girl','cocaine','coke','yayo','blow',
	#	Gun Terms
	'cap' #- bullet.
	'Mac'# 10, Mac - gun.
	'nine', 'nina','9mm'# handgun
	'strapped up',# - holding a gun.
	'AR',#3 - AR 15 machine gun.
	'K','AK',# - ak 47.
	'pump',# - shotgun.
	'pistol',# - gun.
	'ruger',# - gun.
	#Locations
	'areous',# - area.
	'crib',# - place of residence.
	'hood',# - neighbourhood.
	'bodega',# - corner store-wine bar (in Europe).
	
	#Concepts
	'benjamins',# - refers to 100 dollar bills after the American president Benjamin Franklin, whose face is on them. "It's all about the benjamins".

	'bling bling',# - expensive jewelry or other expensive material possessions, refers to the imaginary sound of glistening metal or other shiny surfaces. "bling bling" added to Oxford Dictionary.
	'bo janglin',#–stupid,not paying attention.
	'dime',#-a very attractive woman,a "ten" on a scale of one-to-ten.
	'fly',# - appealing "She is so fly."
	'koolin',# it -kickin' it,chilling.
	'kickin',# - appealing "Those rims are kickin."
	'straight',# - okay.
	'tight',# - attractive "Your car is tight."
	'wack',#-not to one's liking "This relationship is totally wack."
	#Relationships,
	'boo',#- boy/girlfriend, beau.
	'boyz',#- gang friends.
	'brurva',#- a male acquaintance.
	'crew',#- one's good friends (similar to boyz).
	'haps',# mother.
	'hoodrat',# - girl with loose morals.
	'homie', 'homey', 'homeboy' ,#- friend.
	'peeps',# people, acquantainces.
	'shorty',#- girl, or girl on the side.
	

	'jakes',# - cops.
	'jonx', #- belongings.
	'lowckin',# - checking out the scene.
	'sku me',# - excuse me.
	'whas', "goin'",# #down?-what are we doin` tonight?
	'yo','yoo','yoho',#
	'bro','brother',#
	'nigga','ni**a','n****','nigg',#
	'fuck','fucker'#
	
	]
	

	## Full List of Bad Words (Comma-separated-Text-File)
	## =============================================

	## This Full List of Bad Words is provided free by: Free Web Headers ñ www.freewebheaders.com

	## List last updated: Jul 29, 2018

	## URL: https://www.freewebheaders.com/full-list-of-bad-words-banned-by-google/

	## ----------------------------------
	
	f = open(r'dictionary/bad-words.txt','r')
	content = f.read()
	bad_words = content.split(",")

	for indx,w in enumerate(bad_words):
		if w.startswith(' '):
			bad_words[indx] = w.replace(' ','')
			
		elif w.endswith(' '):
			bad_words[indx] = w.replace(' ','')

	word_list_ = hiphop_slang+bad_words

	return word_list_


