## general
import os
from os import listdir, path
import pandas as pd
import numpy as np
import numpy.matlib as matlib

import itertools
from itertools import chain
import json

## colouring
from termcolor import colored

## sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC ## adaboost base model
from sklearn.base import TransformerMixin, BaseEstimator 
from sklearn.preprocessing import StandardScaler

## used for parameter selection
from sklearn import svm,  model_selection # grid search
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

## Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

## ensemble
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

## Principal Component Analyze
from sklearn.decomposition import PCA

## metrics
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

## to save parameters and classifiers
import pickle# as pickle

## Plotting
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import pylab

## for PCA+SVD distance
from scipy.spatial import distance

## debugging
import sys

home_hpc = "/nas/ei/home/ga59qek/FP/"
home_PC = '/Users/guillermoventuramartinezAIR/Desktop/FP/'
## global directories:

home = home_hpc

pl_data = home+'PL_DATA/'

path_RES = home+'user_PL/'

clf_res = home+"CLF_results/"

path_CM = home+'CM_DATA/'



def PL_selector(): 

	uno = ('spotify','Peaceful Piano','37i9dQZF1DX4sWSpwq3LiO')
	
	#tres = ('spotify','mint','37i9dQZF1DX4dyzvuaRJ0n')
	
	tres = ('spotify','All Out 80s','37i9dQZF1DX4UtSsGT1Sbe')

	cinco = ('spotify','Rock Classics','37i9dQZF1DWXRqgorJj26U')
	
	cuatro = ('spotify','¡Viva Latino!','37i9dQZF1DX10zKzsJ2jva')	
	
	#cinco = ('spotify_germany','Pop Remix','37i9dQZF1DXcZDD7cfEKhW')

	dos = ('spotify_germany','Techno Bunker','37i9dQZF1DX6J5NfMJS675')

	#tres = ('spotify_france','Fresh Rap','37i9dQZF1DWU4xkXueiKGW')


	#pl_list = ['Peaceful Piano','Rock Classics','¡Viva Latino!','Techno Bunker','Fresh Rap']
	pl_list = [uno, dos,tres,cuatro,cinco] ##init

	ID = 'EXP_3'
	pca = '' # false
	N_ = [2,3]
	
	X_ALL, y_ALL = get_X_and_y(ID,pl_list,PCA=False)

	X_ALL, X_ALL_val, y_ALL, y_ALL_val = train_test_split(X_ALL,y_ALL,test_size=0.15)
	scaler = StandardScaler()
	X_ALL, X_ALL_val = scale_this(scaler, X_ALL,X_ALL_val)

	save_npy(X_ALL, X_ALL_val, y_ALL, y_ALL_val,'ALL'+pca,'VOTING','EXP3',pl_list)

	for N in N_:
		voting_groups = get_ready(N)
		#print(len(voting_groups))

		for I_, N_group in enumerate(voting_groups):
			print(N_group)
			pipe_list = []
			clf_list = []
			ds_list = []
			for i, _clf_ in enumerate(N_group):
				CLF = _clf_[0]
				DS = _clf_[1]
				print(colored('CLF '+str(i)+':'+CLF+' '+DS,'green'))
				compute_parameters(X_ALL,y_ALL,ID,CLF,DS,pca)
				pipe = train_(X_ALL,y_ALL,X_ALL_val,y_ALL_val,DS,CLF,num_folds=5)
				pipe_list.append(pipe)
				clf_list.append(CLF)
				ds_list.append(DS)

			eclf, score, acc_l, labels, e_val_score,CM,report = the_voting(N,X_ALL,X_ALL_val,y_ALL,y_ALL_val,pipe_list,ds_list,clf_list)
			#print(colored(' VOTING SCORE:'+str(score),'blue'))
			print_save(pl_list,ID, N, labels, acc_l, e_val_score, CM, report,I_)
			CM_plotter(CM,pl_list,ID,N,I_)


	sys.exit('Finished Experiment 3')

def scale_this(scaler, X_train, X_test):
	X_train = scaler.fit_transform(X_train)
	X_test  = scaler.transform(X_test)
	return X_train, X_test

def compute_parameters(X,y,ID,_clf_,DS,pca):

	num_folds=5

	if DS == 'MIR':
		X = X[:,0:34]
	if DS == 'SPO':
		X = X[:,35:47]
	if DS == 'LYR':
		X = X[:,48:95]
	if DS == 'ALL':
		X = X

	num_folds = 5  ##  !! KFOLD SELECT 

	param_folder = home+'parameters/'+ID+'/'
	if not os.path.exists(param_folder):
		os.makedirs(param_folder)

	if not os.path.exists(param_folder+DS+pca+'/'):
		os.makedirs(param_folder+DS+pca+'/')

	param_file = param_folder+'/'+DS+pca+'/'+_clf_+'.pkl'

	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	'+_clf_+' - Parameter Models')
		print('	'+DS+':')
		model_parameters = open_parameters(param_file)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	'+_clf_+' - Parameter Models')
		print('	'+DS+':')
		model_parameters = perform_grid_search(_clf_,X,y,num_folds)
		save_parameters(param_file,model_parameters)	

	#print(model_parameters)
	
def train_(X,y,X_val,y_val,DS,_clf_,num_folds):
	y = y.ravel()

	params = open_parameters(home_hpc+'parameters/EXP_3/'+DS+'/'+_clf_+'.pkl')
	#parameters = param()
	if _clf_ == 'SVM':
		parameters = params()
		model =   svm.SVC(C=parameters['C'],kernel=parameters['kernel'],gamma=parameters['gamma'],degree=parameters['degree'],coef0=parameters['coef0'],shrinking=parameters['shrinking'],tol=parameters['tol'])
	if _clf_ == 'RF':
		parameters = params()
		model = RandomForestClassifier(bootstrap=parameters['bootstrap'],n_estimators=parameters['n_estimators'],max_depth=parameters['max_depth'],max_features=parameters['max_features'],min_samples_leaf = parameters['min_samples_leaf'],min_samples_split = parameters['min_samples_split'],random_state=0,criterion=parameters['criterion'])
	if _clf_ == 'kNN':
		parameters = params()
		model = KNeighborsClassifier(algorithm=parameters['algorithm'],leaf_size=parameters['leaf_size'],metric=parameters['metric'],metric_params=parameters['metric_params'],n_jobs=parameters['n_jobs'],n_neighbors=parameters['n_neighbors'],p=parameters['p'],weights=parameters['weights'])


	if DS == 'MIR': 	## for COLUMN EXTRACTOR CLASS
		cols_ = range(0,34)
	elif DS == 'SPO':
		cols_ = range(35,47)
	elif DS == 'LYR':
		cols_ = range(48,95)
	elif DS == 'ALL':
		cols_ = range(0,95)


	pipe = Pipeline([
		('col_extract', ColumnExtractor( cols=cols_ )), # selecting features 0 and 1 (df1) to be used with  (clf1)
		('clf', model)
		])
	
	print(' -> '+DS+' + '+_clf_)
	pipe.fit(X, y)
	

	print(' Validation:')
	print(pipe.score(X_val,y_val)) # sanity check

	y_true, y_pred = y_val, pipe.predict(X_val)
	cm=confusion_matrix(y_true, y_pred)
	#CM_plotter(cm,pl_list,'EXP3',DS,_clf_)
	print(cm)

	return pipe

def the_voting(N,X_ALL,X_ALL_val,y_ALL,y_ALL_val,pipe_list,DS,CLF):
	y_ALL = y_ALL.ravel()
	y_ALL_val = y_ALL_val.ravel()

	if N==3:
		eclf = VotingClassifier(estimators=[(CLF[0]+'+'+DS[0], pipe_list[0]), (CLF[1]+'+'+DS[1], pipe_list[1]), (CLF[2]+'+'+DS[2],pipe_list[2])], voting='hard',n_jobs=-1)
		print(' Sanity check [Ensemble]')
		print(' -> ALL Together')
		eclf.fit(X_ALL, y_ALL)
		print(eclf.score(X_ALL_val,y_ALL_val))

		y_true, y_pred = y_ALL_val, eclf.predict(X_ALL_val)
		CM=confusion_matrix(y_true, y_pred)
		report = classification_report(y_true, y_pred)
		print(report)
		print(CM)
		print(' --> Validation Accuracy:')
		e_val_score = accuracy_score(y_true, y_pred)
		print('    '+"{:.2%}".format(accuracy_score(y_true, y_pred)))

		acc = []
		labels = []
		## THIS IS FROM sklearn. (sidenote)
		for clf__, label in zip([pipe_list[0], pipe_list[1], pipe_list[2], eclf],[DS[0]+'+'+CLF[0], DS[1]+'+'+CLF[1], DS[2]+'+'+CLF[2], 'Ensemble']):
			scores = cross_val_score(clf__, X_ALL, y_ALL, scoring='accuracy', cv=5)
			print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
			acc.append(scores.mean())
			labels.append(label)

		return eclf, eclf.score(X_ALL_val,y_ALL_val), acc,labels,e_val_score,CM,report

	else:## N==2:
		# ensemble/voting classifier where clf1 fitted with df1 and clf2 fitted with df2
		eclf = VotingClassifier(estimators=[(CLF[0]+'+'+DS[0], pipe_list[0]), (CLF[1]+'+'+DS[1], pipe_list[1])], voting='hard',n_jobs=-1)
		print(' Sanity check [Ensemble]')
		print(' -> ALL Together')
		eclf.fit(X_ALL, y_ALL)
		print(eclf.score(X_ALL_val,y_ALL_val))

		y_true, y_pred = y_ALL_val, eclf.predict(X_ALL_val)
		CM=confusion_matrix(y_true, y_pred)
		report = classification_report(y_true, y_pred)
		print(report)
		print(CM)
		print(' --> Validation Accuracy:')
		e_val_score = accuracy_score(y_true, y_pred)
		print('    '+"{:.2%}".format(accuracy_score(y_true, y_pred)))

		acc = []
		labels = []
		## THIS IS FROM sklearn. (sidenote)
		for clf__, label in zip([pipe_list[0], pipe_list[1], eclf],[DS[0]+'+'+CLF[0], DS[1]+'+'+CLF[1], 'Ensemble']):
			scores = cross_val_score(clf__, X_ALL, y_ALL, scoring='accuracy', cv=5)
			print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
			acc.append(scores.mean())
			labels.append(label)

		return eclf, eclf.score(X_ALL_val,y_ALL_val),acc,labels,e_val_score,CM,report

def get_X_and_y(ID,pl_list,PCA):

	## init X and y 
	y_MIR = [] 
	y_SPO = []
	y_LYR = []

	y_ALL = []

	X_MIR = []
	X_SPO = []
	X_LYR = []

	X_ALL = []

	for index, pl in enumerate(pl_list):

		## unpack tuples
		user = pl[0]
		pl_name = pl[1]
		pl_id = pl[2]


		## open all feature sets:
		f_mir = pl_data+user+'/'+pl_id+'/MIRaudio_features.tsv'
		f_sp  = pl_data+user+'/'+pl_id+'/Spotify_features.tsv'
		f_ly  = pl_data+user+'/'+pl_id+'/Lyrics_features.tsv'
			
		MIR_df = pd.read_csv(f_mir,sep='\t').set_index('Song_ID')
		Sp_df  = pd.read_csv(f_sp,sep='\t').set_index('Song_ID')
		Lyr_df = pd.read_csv(f_ly,sep='\t').set_index('Song_ID')

		## dropping rows with NaN
		MIR_df = MIR_df.dropna()
		Sp_df  = Sp_df.dropna()
		Lyr_df = Lyr_df.dropna()

		## combine feature sets into one
		ALL_df = create_ALL_df(MIR_df,Sp_df,Lyr_df)
		ALL_df = ALL_df.dropna()
		
		print(MIR_df)
		print('MIR-shape:')
		print(MIR_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(Sp_df)
		print('SPO-shape:')
		print(Sp_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(Lyr_df)
		print('LYR-shape')
		print(Lyr_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		print(ALL_df)
		print('ALL-shape')
		print(ALL_df.shape)
		print(' - - - - - - - - - - - - - - - - - - - ')
		

		## DF to NP
		MIR_np = MIR_df.to_numpy()
		Sp_np  = Sp_df.to_numpy()
		Lyr_np = Lyr_df.to_numpy()
		ALL_np = ALL_df.to_numpy()


		## creating X-matrix
		if len(X_MIR)==0: ## init
			X_MIR = MIR_np
			X_SPO = Sp_np
			X_LYR = Lyr_np
			X_ALL = ALL_np

		else:
			X_MIR = np.vstack((X_MIR,MIR_np))
			X_SPO = np.vstack((X_SPO,Sp_np))
			X_LYR = np.vstack((X_LYR,Lyr_np))
			X_ALL = np.vstack((X_ALL,ALL_np))

		## creating y-vector
		## for each feature sets

		## Lyrics-features
		size = Lyr_np.shape
		y_ = [index] * size[0]
		y_LYR.append(y_)

		## Spotify-features
		size = Sp_np.shape
		y_ = [index] * size[0]
		y_SPO.append(y_)

		## MIR-features
		size = MIR_np.shape
		y_ = [index] * size[0]
		y_MIR.append(y_)

		size = ALL_np.shape
		y_ = [index]*size[0]
		y_ALL.append(y_)

	y_MIR = list(chain(*y_MIR))
	y_SPO = list(chain(*y_SPO))
	y_LYR = list(chain(*y_LYR))
	y_ALL = list(chain(*y_ALL))

	y_MIR = np.array([y_MIR]).T
	y_SPO = np.array([y_SPO]).T
	y_LYR = np.array([y_LYR]).T
	y_ALL = np.array([y_ALL]).T

	'''
	## Standard Scaler:
	SS = StandardScaler()
	SS_MIR = SS.fit(X=X_MIR)
	SS_SPO = SS.fit(X=X_SPO)
	SS_LYR = SS.fit(X=X_LYR)
	SS_ALL = SS.fit(X=X_ALL)
	
	X_MIR_ =  SS_MIR.transform()
	X_SPO_ =  SS_SPO.transform(X_SPO)
	X_LYR_ =  SS_LYR.transform(X_LYR)
	X_ALL_ =  SS_ALL.transform(X_ALL)

	X_MIR =  X_MIR_
	X_SPO =	 X_SPO_
	X_LYR =  X_LYR_
	X_ALL =  X_ALL_
	'''
	print("- Shapes MIR: ")
	print(y_MIR.shape)
	print(X_MIR.shape)

	print("- Shapes Spotify: ")
	print(y_SPO.shape)
	print(X_SPO.shape)

	print("- Shapes Lyrics: ")
	print(y_LYR.shape)
	print(X_LYR.shape)

	print("- Shapes Combined Feature Set")
	print(y_ALL.shape)
	print(X_ALL.shape)

	# PCA + SVD 
	if PCA==True:
		print('- reducing dimensions with PCA... ')
		print(' ')
		'''
		print(' -> reducing MIR')
		X_MIR = do_pca(X_MIR)
		print("- Shapes MIR: ")
		print(y_MIR.shape)
		print(X_MIR.shape)
		print(' ')
		print(' -> reducing SPO:')
		X_SPO = do_pca(X_SPO)
		print("- Shapes Spotify: ")
		print(y_SPO.shape)
		print(X_SPO.shape)
		print(' ')
		print(' -> reducing LYR:')
		X_LYR = do_pca(X_LYR)
		print("- Shapes Lyrics: ")
		print(y_LYR.shape)
		print(X_LYR.shape)
		print(' ')
		'''
		print(' -> reducing ALL:')
		X_ALL = do_pca(X_ALL)
		print("- Shapes Combined Feature Set")
		print(y_ALL.shape)
		print(X_ALL.shape)

		pca = 'PCA' 
	else:  ## PCA == False
		pca = '' # whatever...

	#X_MIR, y_MIR
	#X_SPO, y_SPO 
	#X_LYR, y_LYR
	#sys.exit('check')
	print('- Experiment 3 only using the combined Feature Set:')
	return X_ALL, y_ALL

def pre_process(ID, pl_list, PCA):
	## init X and y 
	y_MIR = [] 
	y_SPO = []
	y_LYR = []

	y_ALL = []

	X_MIR = []
	X_SPO = []
	X_LYR = []

	X_ALL = []

	for index, pl in enumerate(pl_list):

		## unpack tuples
		user = pl[0]
		pl_name = pl[1]
		pl_id = pl[2]


		## open all feature sets:
		f_mir = pl_data+user+'/'+pl_id+'/MIRaudio_features.tsv'
		f_sp  = pl_data+user+'/'+pl_id+'/Spotify_features.tsv'
		f_ly  = pl_data+user+'/'+pl_id+'/Lyrics_features.tsv'
			
		MIR_df = pd.read_csv(f_mir,sep='\t').set_index('Song_ID')
		Sp_df  = pd.read_csv(f_sp,sep='\t').set_index('Song_ID')
		Lyr_df = pd.read_csv(f_ly,sep='\t').set_index('Song_ID')

		## combine feature sets into one
		#ALL_df = pd.concat([MIR_df,Sp_df,Lyr_df],axis=1,sort=False)
		ALL_df = MIR_df.join([Sp_df,Lyr_df])

		## dropping rows with NaN
		MIR_df = MIR_df.dropna()
		Sp_df  = Sp_df.dropna()
		Lyr_df = Lyr_df.dropna()

		ALL_df = ALL_df.dropna()

		#print(MIR_df)
		#print(Sp_df)
		#print(Lyr_df)
		#print(ALL_df)

		## DF to NP
		MIR_np = MIR_df.to_numpy()
		Sp_np  = Sp_df.to_numpy()
		Lyr_np = Lyr_df.to_numpy()
		ALL_np = ALL_df.to_numpy()


		## creating X-matrix
		if len(X_MIR)==0: ## init
			X_MIR = MIR_np
			X_SPO = Sp_np
			X_LYR = Lyr_np
			X_ALL = ALL_np

		else:
			X_MIR = np.vstack((X_MIR,MIR_np))
			X_SPO = np.vstack((X_SPO,Sp_np))
			X_LYR = np.vstack((X_LYR,Lyr_np))
			X_ALL = np.vstack((X_ALL,ALL_np))

		## creating y-vector
		## for each feature sets

		## Lyrics-features
		size = Lyr_np.shape
		y_ = [index] * size[0]
		y_LYR.append(y_)

		## Spotify-features
		size = Sp_np.shape
		y_ = [index] * size[0]
		y_SPO.append(y_)

		## MIR-features
		size = MIR_np.shape
		y_ = [index] * size[0]
		y_MIR.append(y_)

		size = ALL_np.shape
		y_ = [index]*size[0]
		y_ALL.append(y_)

	y_MIR = list(chain(*y_MIR))
	y_SPO = list(chain(*y_SPO))
	y_LYR = list(chain(*y_LYR))
	y_ALL = list(chain(*y_ALL))

	y_MIR = np.array([y_MIR]).T
	y_SPO = np.array([y_SPO]).T
	y_LYR = np.array([y_LYR]).T
	y_ALL = np.array([y_ALL]).T



	'''
	## Standard Scaler:
	SS = StandardScaler()
	SS_MIR = SS.fit(X=X_MIR)
	SS_SPO = SS.fit(X=X_SPO)
	SS_LYR = SS.fit(X=X_LYR)
	SS_ALL = SS.fit(X=X_ALL)
	
	X_MIR_ =  SS_MIR.transform()
	X_SPO_ =  SS_SPO.transform(X_SPO)
	X_LYR_ =  SS_LYR.transform(X_LYR)
	X_ALL_ =  SS_ALL.transform(X_ALL)

	X_MIR =  X_MIR_
	X_SPO =	 X_SPO_
	X_LYR =  X_LYR_
	X_ALL =  X_ALL_
	'''
	print("- Shapes MIR: ")
	print(y_MIR.shape)
	print(X_MIR.shape)

	print("- Shapes Spotify: ")
	print(y_SPO.shape)
	print(X_SPO.shape)

	print("- Shapes Lyrics: ")
	print(y_LYR.shape)
	print(X_LYR.shape)

	print("- Shapes Combined Feature Set")
	print(y_ALL.shape)
	print(X_ALL.shape)

	# PCA + SVD 
	if PCA==True:
		print('- reducing dimensions with PCA... ')
		print(' ')
		print(' -> reducing MIR')
		X_MIR = do_pca(X_MIR)
		print("- Shapes MIR: ")
		print(y_MIR.shape)
		print(X_MIR.shape)
		print(' ')
		print(' -> reducing SPO:')
		X_SPO = do_pca(X_SPO)
		print("- Shapes Spotify: ")
		print(y_SPO.shape)
		print(X_SPO.shape)
		print(' ')
		print(' -> reducing LYR:')
		X_LYR = do_pca(X_LYR)
		print("- Shapes Lyrics: ")
		print(y_LYR.shape)
		print(X_LYR.shape)
		print(' ')
		print(' -> reducing ALL:')
		X_ALL = do_pca(X_ALL)
		print("- Shapes Combined Feature Set")
		print(y_ALL.shape)
		print(X_ALL.shape)


		pca = 'PCA'
	else:  ## PCA == False
		pca = ''

	## SPLIT FOR VALIDATION:
	
	X_MIR, X_MIR_val, y_MIR, y_MIR_val = train_test_split(X_MIR, y_MIR, test_size=0.15,random_state=42)
	X_SPO, X_SPO_val, y_SPO, y_SPO_val = train_test_split(X_SPO, y_SPO, test_size=0.15,random_state=42)
	X_LYR, X_LYR_val, y_LYR, y_LYR_val = train_test_split(X_LYR, y_LYR, test_size=0.15,random_state=42)
	X_ALL, X_ALL_val, y_ALL, y_ALL_val = train_test_split(X_ALL, y_ALL, test_size=0.15,random_state=42)

	## packing for reading:

	MIR_DS = [X_MIR, X_MIR_val, y_MIR, y_MIR_val]
	SPO_DS = [X_SPO, X_SPO_val, y_SPO, y_SPO_val]
	LYR_DS = [X_LYR, X_LYR_val, y_LYR, y_LYR_val]
	ALL_DS = [X_ALL, X_ALL_val, y_ALL, y_ALL_val]

	print(' EXPERIMENT 3.')
	return MIR_DS,SPO_DS,LYR_DS,ALL_DS

def get_ready(N):
	return_L = []

	## debugging:
	clf_L =  ['SVM','kNN','RF']
	DS_L = ['MIR','SPO','LYR']#,'MIRPCA','SPOPCA','LYRPCA']#,'ALL']

	C = list(itertools.product(clf_L,DS_L))
	C_= list(itertools.combinations(C,N))			# N = 3 // 2


	for i,e in enumerate(C_):

		#print(e)
		#print(' - CLF1:')
		#print(' ... DS:')
		DS1 = e[0][1]
		#print(e[0][1])
		#print(' ... CLF:')
		CLF1 = e[0][0]
		#print(e[0][0])
		var_C1 = [DS1,CLF1]

		#print(' - CLF2:')
		#print(' ... DS:')
		DS2 = e[1][1]
		#print(e[1][1])
		#print(' ... CLF:')
		CLF2 = e[1][0]
		#print(e[1][0])
		var_C2 = [DS2,CLF2]

		if N==3:
			#print(' - CLF3:')
			#print(' ... DS:')
			DS3 = e[2][1]
			#print(e[2][1])
			#print(' ... CLF:')
			CLF3 = e[2][0]
			#print(e[2][0])
			var_C3 = [DS3,CLF3]

		return_L.append(e)

	return return_L

def get_X_and_y_Voting(N, pl_indx ,MIR_DS, SPO_DS, LYR_DS, ALL_DS, pl_list):
	## debugging:
	clf_L =  ['SVM','kNN','RF']
	DS_L = ['MIR','SPO','LYR']#,'MIRPCA','SPOPCA','LYRPCA']#,'ALL']

	C = list(itertools.product(clf_L,DS_L))
	C_= list(itertools.combinations(C,N))			# N = 3 // 2


	for i,e in enumerate(C_):

		print(e)
		print(' - CLF1:')
		print(' ... DS:')
		DS1 = e[0][1]
		print(e[0][1])
		print(' ... CLF:')
		CLF1 = e[0][0]
		print(e[0][0])
		var_C1 = [DS1,CLF1]

		print(' - CLF2:')
		print(' ... DS:')
		DS2 = e[1][1]
		print(e[1][1])
		print(' ... CLF:')
		CLF2 = e[1][0]
		print(e[1][0])
		var_C2 = [DS2,CLF2]

		if N==3:
			print(' - CLF3:')
			print(' ... DS:')
			DS3 = e[2][1]
			print(e[2][1])
			print(' ... CLF:')
			CLF3 = e[2][0]
			print(e[2][0])
			var_C3 = [DS3,CLF3]


		X_MIR, X_MIR_val, y_MIR, y_MIR_val = MIR_DS[0],MIR_DS[1],MIR_DS[2],MIR_DS[3]
		X_SPO, X_SPO_val, y_SPO, y_SPO_val = SPO_DS[0],SPO_DS[1],SPO_DS[2],SPO_DS[3]
		X_LYR, X_LYR_val, y_LYR, y_LYR_val = LYR_DS[0],LYR_DS[1],LYR_DS[2],LYR_DS[3]
		X_ALL, X_ALL_val, y_ALL, y_ALL_val = ALL_DS[0],ALL_DS[1],ALL_DS[2],ALL_DS[3]

		## load matrices:
		X_train_C1, y_train_C1, X_val_C1, y_val_C1 = load_matrices('EXP_3',CLF1,DS1)
		#X_all_C1, y_all_C1, X_all_val_C1,y_all_val_C1 = load_matrices('EXP_3',CLF1,'ALL')
		X_train_C2, y_train_C2, X_val_C2, y_val_C2 = load_matrices('EXP_3',CLF2,DS2)
		#X_all_C2, y_all_C2, X_all_val_C2,y_all_val_C2 = load_matrices('EXP_3',CLF2,'ALL')

		if N==3:
			X_train_C3, y_train_C3, X_val_C3, y_val_C3 = load_matrices('EXP_3',CLF3,DS3)
		#X_all_C3, y_all_C3, X_all_val_C3,y_all_val_C3 = load_matrices('EXP_3',CLF3,'ALL')


		clf_1 = load_classifier(DS1,CLF1,pl_indx)
		clf_2 = load_classifier(DS2,CLF2,pl_indx)
		## 
		## 
		
		if N==3:
			clf_3 = load_classifier(DS3,CLF3,pl_indx)
			## EnsembleVoting :
			eclf,score = ensamble_voting_pipeline(N,X_ALL,X_ALL_val,y_ALL,y_ALL_val,clf_1,clf_2,clf_3,DS1,DS2,DS3,CLF1,CLF2,CLF3)
		else: ## N==2:
			eclf,score = ensamble_voting_pipeline(N,X_ALL,X_ALL_val,y_ALL,y_ALL_val,clf_1,clf_2,0,DS1,DS2,0,CLF1,CLF2,0)

		## VALIDATION SET 1
		print(' -------------------------- ')
		print(' Pediction Validation  Set:')
		print(' VALIDATION ACC: '+str(score))
		y_true, y_pred = y_ALL_val, eclf.predict(X_ALL_val)
		cm = confusion_matrix(y_true,y_pred)
		print(cm)
		CM_plotter(cm,pl_list,'EXP3',N,e)

		#print_save(pl_indx,pl_list,data_set, clf, y_true, y_pred, X_val, y_val, params)
'''
def CM_plotter(CM,pl_list,ID,N,_clfs_):
	target_names = []
	for pl in pl_list:
		target_names.append(pl[1])

	# Normalise
	cmn = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots(figsize=(5,5))
	sns.heatmap(cmn, annot=True, fmt='.2f', yticklabels=target_names)
	plt.ylabel('Actual')
	plt.xlabel('Predicted')
	#plt.xticks(rotation=45) 
	if N == 2:
		plt.savefig(path_CM+ID+' N='+N+" "+_clfs_[0][0]+'-'+_clfs_[0][1]+'&'+_clfs_[1][0]+'-'+_clfs_[1][1]+'.eps')
	else:
		plt.savefig(path_CM+ID+' N='+N+" "+_clfs_[0][0]+'-'+_clfs_[0][1]+'&'+_clfs_[1][0]+'-'+_clfs_[1][1]+'&'+_clfs_[2][0]+'-'+_clfs_[2][1]+'.eps')
'''

def ensamble_voting_pipeline(N,X_train,X_test,y_train,y_test,clf_1,clf_2,clf_3,DS1,DS2,DS3,CLF1,CLF2,CLF3):
	y_train = y_train.ravel()
	y_test = y_test.ravel()

	if DS1 == 'MIR': 	## for COLUMN EXTRACTOR CLASS
		cols_1 = range(0,34)
	elif DS1 == 'SPO':
		cols_1 = range(35,47)
	elif DS1 == 'LYR':
		cols_1 = range(48,95)

	if DS2 == 'MIR':
		cols_2 = range(0,34)
	elif DS2 == 'SPO':
		cols_2 = range(35,47)
	elif DS2 == 'LYR':
		cols_2 = range(48,95)

	if N==3:
		if DS3 == 'MIR':
			cols_3 = range(0,34)
		elif DS3 == 'SPO':
			cols_3 = range(35,47)
		elif DS3 == 'LYR':
			cols_3 = range(48,95)


	######################
	# fit clf1 with df1
	pipe1 = Pipeline([
		('col_extract', ColumnExtractor( cols=cols_1 )), # selecting features 0 and 1 (df1) to be used with  (clf1)
		('clf', clf_1)
		])
	print(' Sanity check')
	print(' -> '+DS1+' + '+CLF1)
	pipe1.fit(X_train, y_train) # sanity check
	
	print(pipe1.score(X_test,y_test)) # sanity check
	

	######################
	# fit clf2 with df2
	pipe2 = Pipeline([
		('col_extract', ColumnExtractor( cols=cols_2)), # selecting features 2 and 3 (df2) to be used with  (clf2)
		('clf', clf_2)
		])

	print(' Sanity check')
	print(' -> '+DS2+' + '+CLF2)
	pipe2.fit(X_train, y_train) # sanity check
	print(pipe2.score(X_test,y_test)) # sanity check
	
	
	if N==3:
		######################
		# fit clf3 with df3
		pipe3 = Pipeline([
			('col_extract', ColumnExtractor( cols=cols_3)), # selecting features 2 and 3 (df2) to be used with SVC (clf2)
			('clf', clf_3)
			])

		print(' Sanity check')
		print(' -> '+DS3+' + '+CLF3)
		pipe3.fit(X_train, y_train) # sanity check
		print(pipe3.score(X_test,y_test)) # sanity check

		######################
		# ensemble/voting classifier where clf1 fitted with df1 and clf2 fitted with df2
		eclf = VotingClassifier(estimators=[(CLF1+'+'+DS1, pipe1), (CLF2+'+'+DS2, pipe2), (CLF3+'+'+DS3,pipe3)], voting='hard',n_jobs=-1)
		print(' Sanity check [Ensemble]')
		print(' -> ALL Together')
		eclf.fit(X_train, y_train)
		print(eclf.score(X_test,y_test))

		## THIS IS FROM sklearn. (sidenote)
		for clf__, label in zip([clf_1, clf_2, clf_3, eclf],[DS1+'+'+CLF1, DS2+'+'+CLF2, DS3+'+'+CLF3, 'Ensemble']):
			#print(label)
			#print(label.split('-')[0])
			#print(label.split('-')[1])
			#DS__ = label.split('-')[0]
			#CLF__= label.split('-')[1]
			#print(DS__)
			#print(CLF__)
			#if 
			#sys.exit('wait')
			scores = cross_val_score(clf__, X_train, y_train, scoring='accuracy', cv=5)
			print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

		return eclf, eclf.score(X_test,y_test)
	else:## N==2:
		# ensemble/voting classifier where clf1 fitted with df1 and clf2 fitted with df2
		eclf = VotingClassifier(estimators=[(CLF1+'+'+DS1, pipe1), (CLF2+'+'+DS2, pipe2)], voting='hard',n_jobs=-1)
		print(' Sanity check [Ensemble]')
		print(' -> ALL Together')
		eclf.fit(X_train, y_train)
		print(eclf.score(X_test,y_test))

		## THIS IS FROM sklearn. (sidenote)
		for clf__, label in zip([clf_1, clf_2, eclf],[DS1+'+'+CLF1, DS2+'+'+CLF2, 'Ensemble']):
			#print(label)
			#print(label.split('-')[0])
			#print(label.split('-')[1])
			#DS__ = label.split('-')[0]
			#CLF__= label.split('-')[1]
			#print(DS__)
			#print(CLF__)
			#if 
			#sys.exit('wait')
			scores = cross_val_score(clf__, X_train, y_train, scoring='accuracy', cv=5)
			print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

		return eclf, eclf.score(X_test,y_test)

######################
# custom transformer for sklearn pipeline
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[:, c:c+1])
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self

def load_classifier(dataset,clf,pl_indx):

	clf_dir = home+'/classifiers/'+str(pl_indx)+'/'+dataset+'/'

	#clf_list = os.listdir(clf_dir)

	clf_file = clf_dir+clf+'.pkl'

	with open(clf_file,"rb") as file:
		clf_res = pickle.load(file)

	return clf_res

def save_parameters(param_file,model_parameters):
	
	with open(param_file,"wb") as file:
		pickle.dump(model_parameters, file, pickle.HIGHEST_PROTOCOL)
	
#									MODEL = SVM/RF/kNN
def save_npy(X,X_val,y,y_val,DataSet,MODEL,pl_indx,classification_set):
	print('- saving Training/Validation Matrixes:')
	print(pl_indx)
	print(classification_set)
	print(DataSet)
	print(MODEL)
	path__ = home+'/SETS/'+str(pl_indx)+'/'
	if not os.path.exists(path__):
		os.makedirs(path__)

	path__ = path__+MODEL+'/'
	if not os.path.exists(path__):
		os.makedirs(path__)

	path__DS = path__+DataSet+'/'
	if not os.path.exists(path__DS):
		os.makedirs(path__DS)
	
	path__T = path__DS+'Training/'
	path__V = path__DS+'Validation/'
	
	if not os.path.exists(path__T):
		os.makedirs(path__T)
	
	if not os.path.exists(path__V):
		os.makedirs(path__V)


	## THINGS TO SAVE: X,X_val,y,y_val
	## save Training 
	file = path__T+'X_train'
	np.save(file, X, allow_pickle=True)

	file = path__T+'y_train'
	np.save(file, y, allow_pickle=True)

	file = path__V+'X_val'
	np.save(file, X_val, allow_pickle=True)

	file = path__V+'y_val'
	np.save(file, y_val, allow_pickle=True)

def open_parameters(param_file):
	#	'/Users/guillermoventuramartinezAIR/Desktop/FP/parameters/PL_Set_'+str(pl_indx)+'/'
	#param_file = home+'parameters/EX1_'+str(pl_indx)+'/'+dataset+'/'+clf+'.pkl'
	
	file = open(param_file, 'rb')
	parameters = pickle.load(file)
	file.close()
	
	return parameters

def save_clf(clf,clf_type,dataset,pl_indx):

	if not path.exists(home+'classifiers/'+str(pl_indx)+'/'):
		os.mkdir(home+'/classifiers/'+str(pl_indx)+'/')	

	clf_dir = home+'classifiers/'+str(pl_indx)+'/'+dataset+'/'
	if not path.exists(clf_dir):
		os.mkdir(clf_dir)

	clf_file = clf_dir+clf_type+'.pkl'

	with open(clf_file,"wb") as file:
		pickle.dump(clf, file, pickle.HIGHEST_PROTOCOL)

def CM_plotter(CM,pl_list,ID,N_,I_):
	if not path.exists(path_CM+'/'+ID+'/'):
		os.mkdir(path_CM+'/'+ID+'/')
	if not path.exists(path_CM+'/'+ID+'/N='+str(N_)+'/'):
		os.mkdir(path_CM+'/'+ID+'/N='+str(N_)+'/')

	target_names = []
	for pl in pl_list:
		target_names.append(pl[1])

	# Normalise
	cmn = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots(figsize=(5,5))
	sns.heatmap(cmn, annot=True, fmt='.2f', yticklabels=target_names)
	plt.ylabel('Actual')
	plt.xlabel('Predicted')
	#plt.xticks(rotation=45) 
	plt.savefig(path_CM+'/'+ID+'/N='+str(N_)+'/'+'#'+str(I_)+'.eps',bbox_inches='tight')

def print_save(pl_list,ID, N, labels, acc_train, acc_validation, CM, report,I_):
	## ID N pl_list ensemble_list_tupl
	## to do parameters 
	## votingclassifier.estimators_ + votingclassifier.named_estimators

	print(" # * # * # * # * # * # * # * #")
	print(' - Print-save Experiment 3')
	print(" # * # * # * # * # * # * # * #")

	parent_folder = home+"CLF_results/Exp_3/"
	if not path.exists(parent_folder):
		os.mkdir(parent_folder)

	folder = parent_folder+"N_"+str(N)+"/"
	if not path.exists(folder):
		os.mkdir(folder)

	list_clf =os.listdir(folder)
	
	file = folder+'Voting_Classifier.txt'

	text_file = open(file, "a+")
	text_file.write(" # * # * # * # * # * # * # * #\n")
	text_file.write(" - CLF trained with\n")
	for i, pl in enumerate(pl_list):
		text_file.write(' - ############:\t'+str(i)+"\n")
		text_file.write(' - USER:\t'+str(pl[0])+"\n")
		text_file.write(' - PLAYLIST:\t'+str(pl[1])+"\n")
		text_file.write(' - PL ID\t'+str(pl[2])+"\n")
		text_file.write(" - - - - - - - - - - - - - - -\n")
	text_file.write(' --> Number:'+str(I_))
	text_file.write(" -------- Trained CLF --------\n")
	text_file.write(' CLF 1:\t'+labels[0]+'\t'+'--> Accuracy: '+str(acc_train[0])+'\n')
	text_file.write(' CLF 2:\t'+labels[1]+'\t'+'--> Accuracy: '+str(acc_train[1])+'\n')
	if N == 3:
		text_file.write(' CLF 3:\t'+labels[2]+'\t'+'--> Accuracy: '+str(acc_train[2])+'\n')
	text_file.write(' \n') # empty line
	text_file.write(' Ensemble Voting:\t '+ '--> Accuracy: '+str(acc_train[-1])+'\n')
	text_file.write(' \t--> Validation Accuracy: '+str(acc_validation))
	text_file.write('\n')
	text_file.write(report)
	text_file.write('\n')
	text_file.write(np.array2string(CM))
	text_file.write('\n')
	text_file.write(" - - - - - - - - - - - ")
	text_file.write('\n')
	text_file.write(" # * # * # * # * # * # * # * #")
	text_file.write('\n')
	text_file.close()

def create_ALL_df(MIR_df,Sp_df,Lyr_df):

	#L = [MIR_df,Sp_df,Lyr_df]
	ALL_df = MIR_df.join([Sp_df,Lyr_df])

	#ALL_df = pd.concat(L,axis=1,sort=False).set_index('Song_ID')
	#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
	#	print(ALL_df)
	
	return ALL_df

def perform_grid_search(model, X, y, num_folds):
	y = np.ravel(y)

	# Split the dataset in two equal parts
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)

	if model == 'SVM':
		model = SVC()

		# parameters:
		param_grid_ = [{'kernel': ['rbf'], 'gamma': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100,1000,10000],'C': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100,1000,10000]},
						{'kernel': ['sigmoid'],'gamma': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100,1000,10000], 'C': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100,1000,10000]}]
		
	elif model == "RF":
		model = RandomForestClassifier()

		# parameters:
		param_grid_ = {'bootstrap': [True],
					'max_depth': [1,10,20,50,100,100],#[1,10,100],
					'max_features': [2],#,3],
					'min_samples_leaf': [3, 4, 5],
					'min_samples_split': [8, 10, 12],
					'n_estimators': [1,10,50,100, 200, 500]#'n_estimators': [1,10] ## leaving this one out for now 
					}

	elif model == 'kNN':
		model = KNeighborsClassifier()

		# parameters:
		param_grid_ = {'n_neighbors': [3,4,5,6,7,5,10,15,20],
					'weights': ["uniform","distance"],
					'algorithm': ['auto'],#'ball_tree','kd_tree','brute'],
					'leaf_size': [1,5,10,20,30,40,50,100],#,30],
					'p':[1,2],
					'n_jobs':[-1]
					}
		##

	grid_search_ = GridSearchCV(model, param_grid=param_grid_, cv=num_folds, n_jobs=-1)
	grid_search_.fit(X_train, y_train)

	print("Best parameters set found on development set:")
	print()
	print(grid_search_.best_params_)
	print("{:.2%}".format(grid_search_.best_score_))
	print()
	
	##'''
	print(' -- -- -- -- -- -- -- -- -- -- -- ')
	print('')
	print("Detailed classification report:")
	print("	- Grid scores on development set:")
	print()
	means = grid_search_.cv_results_['mean_test_score']
	stds = grid_search_.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, grid_search_.cv_results_['params']):
		print("- mean: %0.3f \n - std: (+/-%0.03f) \n - parameters: %r"% (mean, std * 2, params))
		print(' * * * * * * * * * * * * * * *')
	print('')
	print(' -- -- -- -- -- -- -- -- -- -- -- ')
	print('')
	##'''

	best_clf = grid_search_.best_estimator_
	y_true, y_pred = y_test, best_clf.predict(X_test)

	print(' --> Validation: ')
	print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred))
	print(confusion_matrix(y_true,y_pred))
	print()

	res_parameters = best_clf.get_params
	
	print(res_parameters)

	return res_parameters

PL_selector()

