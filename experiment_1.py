## general
import os
from os import listdir, path
import glob

## data
import pandas as pd
import numpy as np
import numpy.matlib as matlib
from itertools import chain
import json

## plotting
import matplotlib.pyplot as plt

## Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

## pre-process
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

## coloring
from termcolor import colored

## metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

## Principal Component Analyze
from sklearn.decomposition import PCA

## to save parameters and classifiers
import pickle# as pickle
## for PCA+SVD distance
from scipy.spatial import distance

## debugging
import sys


# directories
home_hpc = "/nas/ei/home/ga59qek/FP/"
home_PC = '/Users/guillermoventuramartinezAIR/Desktop/FP/'
home_work = '...'# variates

home = home_hpc

pl_data = home+'PL_DATA/'

path_RES = home+'user_PL/'

clf_res = home+"CLF_results/"


clf_ = ['SVM','RF','kNN']

def init():

	PL_selector()
	
	for _clf_ in clf_:
		get_res(_clf_)



def PL_selector():	## selects the playlist out of the info file. 

	uno = ('spotify','Peaceful Piano','37i9dQZF1DX4sWSpwq3LiO')
	dos = ('spotify','mint','37i9dQZF1DX4dyzvuaRJ0n')
	tres = ('spotify','Rock Classics','37i9dQZF1DWXRqgorJj26U')
	cuatro = ('spotify','¡Viva Latino!','37i9dQZF1DX10zKzsJ2jva')	
	cinco = ('spotify_germany','Pop Remix','37i9dQZF1DXcZDD7cfEKhW')
	seis = ('spotify_germany','Techno Bunker','37i9dQZF1DX6J5NfMJS675')
	siete = ('spotify_france','Fresh Rap','37i9dQZF1DWU4xkXueiKGW')
	ocho = ('spotify_uk_','Massive Dance Hits','37i9dQZF1DX1N5uK98ms5p')
	nueve = ('spotify_uk_', 'Yoga & Meditation', '37i9dQZF1DX9uKNf5jGX6m')
	#once = ('chillhopmusic','lofi hip hop beats - music to study/relax to ( lo-fi chill hop )','74sUjcvpGfdOvCHvgzNEDO')
	once = ('nme.com','50 of the greatest ever indie breakup songs','6gihjAGHPlx1JrZ5Dg1wuy')
	#doce = ('spotifyusa','Wake Up','3uz1GV5nKYM4XPhBJagqgy')
	diez = ('spotify_espa%C3%B1a','mint Latin','37i9dQZF1DX07RJCJCOYpi')
	
	pl_sets = [uno, dos] ##init
	all__ = [tres,cuatro,cinco,seis,siete,ocho,nueve,diez,once]#,once,doce,trece]

	## comment -> in order to re-compute:	
	#del_previous_results()

	num_folds = 5
	PCA = True

	if PCA == True:
		pca = 'PCA'
	else:
		pca = ''

	pl_indx = 2
	while len(all__)!= 0:
		
		print(' - - - - - - - - - - - - - - - - ')
		print(pl_sets)
		print(' - - - - - - - - - - - - - - - - ')

		## get matrices
		X_MIR, y_MIR, X_SPO, y_SPO, X_LYR, y_LYR, X_ALL, y_ALL = get_X_and_y('EXP1',pl_sets,PCA)
		
		## pre-processing
		X_MIR, X_MIR_val, y_MIR, y_MIR_val, X_SPO, X_SPO_val, y_SPO, y_SPO_val, X_LYR, X_LYR_val, y_LYR, y_LYR_val, X_ALL, X_ALL_val, y_ALL, y_ALL_val = pre_processing(PCA,X_MIR, y_MIR, X_SPO, y_SPO, X_LYR, y_LYR, X_ALL, y_ALL, pl_sets, pl_indx)
		
		for _clf_ in clf_: # clf_ = ['SVM','RF','kNN']
			param_MIR, param_SPO, param_LYR, param_ALL = get_parameters_(pl_indx,_clf_,PCA,num_folds,X_MIR, y_MIR, X_SPO, y_SPO, X_LYR, y_LYR, X_ALL, y_ALL)
			
			print('\t'+colored(_clf_,'cyan')+'\t'+colored('MIR','blue'))
			clf, score = train_(_clf_,X_MIR,y_MIR,num_folds,param_MIR)
			save_clf(clf,_clf_,'MIR'+pca,pl_indx)
			print(" - VALIDATION: ")
			print('')
			y_true, y_pred = y_MIR_val, clf.predict(X_MIR_val)
			print_save(pl_indx,pl_sets,'MIR DATASET '+pca, _clf_ ,y_true, y_pred, X_MIR_val, y_MIR_val, param_MIR)
			
			print('\t'+colored(_clf_,'cyan')+'\t'+colored('SPO','blue'))
			clf, score = train_(_clf_,X_SPO,y_SPO,num_folds,param_SPO)
			save_clf(clf,_clf_,'SPO'+pca,pl_indx)
			print(" - VALIDATION: ")
			print('')
			y_true, y_pred = y_SPO_val, clf.predict(X_SPO_val)
			print_save(pl_indx,pl_sets,'SPO DATASET '+pca, _clf_ ,y_true, y_pred, X_SPO_val, y_SPO_val, param_SPO)

			print('\t'+colored(_clf_,'cyan')+'\t'+colored('LYR','blue'))
			clf, score = train_(_clf_,X_LYR,y_LYR,num_folds,param_LYR)
			save_clf(clf,_clf_,'LYR'+pca,pl_indx)
			print(" - VALIDATION: ")
			print('')
			y_true, y_pred = y_LYR_val, clf.predict(X_LYR_val)
			print_save(pl_indx,pl_sets,'LYR DATASET '+pca, _clf_ ,y_true, y_pred, X_LYR_val, y_LYR_val, param_LYR)

			print('\t'+colored(_clf_,'cyan')+'\t'+colored('ALL','blue'))
			clf, score = train_(_clf_,X_ALL,y_ALL,num_folds,param_ALL)
			save_clf(clf,_clf_,'ALL'+pca,pl_indx)
			print(" - VALIDATION: ")
			print('')
			y_true, y_pred = y_ALL_val, clf.predict(X_ALL_val)
			print_save(pl_indx,pl_sets,'ALL DATASET '+pca, _clf_ ,y_true, y_pred, X_ALL_val, y_ALL_val, param_ALL)

		#get_X_and_y_SVM(pl_indx, pl_sets,PCA=True)
		#get_X_and_y_RF(pl_indx, pl_sets,PCA=True)
		#get_X_and_y_kNN(pl_indx, pl_sets,PCA=True)

		# next iteration:	
		pl_sets.insert(len(pl_sets),all__[0])		
		all__.pop(0)
		pl_indx+=1

def train_(MODEL,X,y,num_folds,params):

	parameters 	= params()
	if MODEL=='SVM':
		clf = SVC(C=parameters['C'],kernel=parameters['kernel'],gamma=parameters['gamma'], probability=True)
	elif MODEL =='RF':
		clf = RandomForestClassifier(bootstrap=parameters['bootstrap'],n_estimators=parameters['n_estimators'],max_depth=parameters['max_depth'],max_features=parameters['max_features'],min_samples_leaf = parameters['min_samples_leaf'],min_samples_split = parameters['min_samples_split'],random_state=0,criterion=parameters['criterion'])
	elif MODEL == 'kNN':
		clf = KNeighborsClassifier(n_neighbors=parameters['n_neighbors'],weights=parameters['weights'],algorithm=parameters['algorithm'],leaf_size=parameters['leaf_size'],p=parameters['p'],n_jobs=parameters['n_jobs'])

	clfs = []  ## for best selection
	cms = []	## multiple confussion matrixes
	
	scores = [] 	## for the results
	
	train_errors = []
	test_errors = []
	training_scores = []
	testing_scores = []

	y = y.ravel()

	kf = KFold(n_splits= num_folds ,shuffle=True)

	for train_index, test_index in kf.split(X):
		#print("TRAIN:\n", train_index)
		#print("TEST:\n", test_index)
		#print(type(train_index))
		#print(type(test_index))
		print('Training kFold')

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		clf.fit(X_train, y_train)

		train_score = clf.score(X_train, y_train)
		test_score = clf.score(X_test, y_test)

		scores.append(test_score)

		training_scores.append(train_score)
		testing_scores.append(test_score)

		train_errors.append(1 - train_score)
		test_errors.append(1 - test_score)

		y_pred = clf.predict(X_test)

		cm = confusion_matrix(y_test,y_pred)

		clfs.append(clf)

		cms.append(cm)

	## score of best clf
	best_score, indx = max((val, idx) for (idx, val) in enumerate(scores))
	
	best_clf = clfs[indx]

	#best_clf.get_params()

	
	print(" - - - - - - - - - - - ")
	print("CONFUSSION MATRIX:")
	print(np.asarray(cms))
	print(" - - - - - - - - - - - ")
	print(" - - - - - - - - - - - ")
	print(" BEST CLASSIFICATION SCORE:")
	print("	"+str(best_score*100)+"%")
	print(" - - - - - - - - - - - ")
	print("")
	print("SCORES IN TRAINING: " )
	print("mean ", np.mean(training_scores))
	print("std  ", np.std(training_scores))
	print(training_scores)
	print(" - - - - - - - - - - - ")
	print("Min Test Errors: ",min(test_errors))
	print(" - - - - - - - - - - - ")	
	print("SCORES IN TEST: "  )
	print("mean ",np.mean(testing_scores))
	print("std  ",np.std(testing_scores))
	print(testing_scores)
	print(" - - - - - - - - - - - ")
	print("TRAINING Parameters")
	print(" # of folds : ", num_folds)
	print(" PARAMETERS of "+MODEL+":")
	print(best_clf.get_params())
	print(" - - - - - - - - - - - ")
	print('')

	return best_clf, best_score

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

		## ALL-features	
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
		print(' ')

		pca = 'PCA' 
	else:  ## PCA == False
		pca = '' # whatever...

	return X_MIR, y_MIR, X_SPO, y_SPO, X_LYR, y_LYR, X_ALL, y_ALL

def pre_processing(PCA,X_MIR, y_MIR, X_SPO, y_SPO, X_LYR, y_LYR, X_ALL, y_ALL,pl_list,pl_indx):
	
	if PCA==True:
		pca='pca'
	else:
		pca=''
	
	## SPLIT FOR VALIDATION:
	X_MIR, X_MIR_val, y_MIR, y_MIR_val = train_test_split(X_MIR, y_MIR, test_size=0.15,random_state=42)
	X_SPO, X_SPO_val, y_SPO, y_SPO_val = train_test_split(X_SPO, y_SPO, test_size=0.15,random_state=42)
	X_LYR, X_LYR_val, y_LYR, y_LYR_val = train_test_split(X_LYR, y_LYR, test_size=0.15,random_state=42)
	X_ALL, X_ALL_val, y_ALL, y_ALL_val = train_test_split(X_ALL, y_ALL, test_size=0.15,random_state=42)
	
	## choose a scaler
	ss = StandardScaler()
	X_MIR,X_MIR_val = scale_this(ss, X_MIR, X_MIR_val)
	X_SPO,X_SPO_val = scale_this(ss, X_SPO, X_SPO_val)
	X_LYR,X_LYR_val = scale_this(ss, X_LYR, X_LYR_val)
	X_ALL,X_ALL_val = scale_this(ss, X_ALL, X_ALL_val)

	## save matrices for later just in case	
	save_npy(X_MIR, X_MIR_val, y_MIR, y_MIR_val,'MIR'+pca,pl_indx,pl_list)
	save_npy(X_SPO, X_SPO_val, y_SPO, y_SPO_val,'SPO'+pca,pl_indx,pl_list)
	save_npy(X_LYR, X_LYR_val, y_LYR, y_LYR_val,'LYR'+pca,pl_indx,pl_list)
	save_npy(X_ALL, X_ALL_val, y_ALL, y_ALL_val,'ALL'+pca,pl_indx,pl_list)

	return X_MIR, X_MIR_val, y_MIR, y_MIR_val, X_SPO, X_SPO_val, y_SPO, y_SPO_val, X_LYR, X_LYR_val, y_LYR, y_LYR_val, X_ALL, X_ALL_val, y_ALL, y_ALL_val

def scale_this(scaler, X_train, X_test):
	X_train = scaler.fit_transform(X_train)
	X_test  = scaler.transform(X_test)
	return X_train, X_test

def get_parameters_(pl_indx,MODEL,PCA,num_folds,X_MIR, y_MIR, X_SPO, y_SPO, X_LYR, y_LYR, X_ALL, y_ALL):
	if PCA==True:
		pca='PCA'
	else:
		pca='' 

	param_folder = home+'parameters/EXP1/'
	if not os.path.exists(param_folder):
		os.makedirs(param_folder)

	param_folder = param_folder+'PL_'+str(pl_indx)+'/'
	if not os.path.exists(param_folder):
		os.makedirs(param_folder)

	if not os.path.exists(param_folder+'MIR'+pca+'/'):
		os.makedirs(param_folder+'MIR'+pca+'/')
	if not os.path.exists(param_folder+'SPO'+pca+'/'):
		os.makedirs(param_folder+'SPO'+pca+'/')
	if not os.path.exists(param_folder+'LYR'+pca+'/'):
		os.makedirs(param_folder+'LYR'+pca+'/')
	if not os.path.exists(param_folder+'ALL'+pca+'/'):
		os.makedirs(param_folder+'ALL'+pca+'/')

	## - MIR 
	param_file = param_folder+'/MIR'+pca+'/'+MODEL+'.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	'+MODEL+' - Parameter Models')
		print('	MIR:')
		model_parameters_MIR = open_parameters(param_file)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	'+MODEL+' - Parameter Models')
		print('	MIR:')
		model_parameters_MIR = perform_grid_search(MODEL,X_MIR,y_MIR,num_folds)
		save_parameters(param_file,model_parameters_MIR)	

	## - SPO
	param_file = param_folder+'/SPO'+pca+'/'+MODEL+'.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	'+MODEL+' - Parameter Models')
		print('	SPO:')
		model_parameters_SPO = open_parameters(param_file)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	'+MODEL+' - Parameter Models')
		print('	SPO:')
		model_parameters_SPO = perform_grid_search(MODEL,X_SPO,y_SPO,num_folds)
		save_parameters(param_file,model_parameters_SPO)		

	## - LYR
	param_file = param_folder+'/LYR'+pca+'/'+MODEL+'.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print(' '+MODEL+' - Parameter Models')
		print('	LYR:')
		model_parameters_LYR = open_parameters(param_file)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	'+MODEL+' - Parameter Models')
		print('	LYR:')
		model_parameters_LYR = perform_grid_search(MODEL,X_LYR,y_LYR,num_folds)
		save_parameters(param_file,model_parameters_LYR)			

	## - ALL 
	param_file = param_folder+'/ALL'+pca+'/'+MODEL+'.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	'+MODEL+' - Parameter Models')
		print('	ALL:')
		model_parameters_ALL = open_parameters(param_file)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	'+MODEL+' - Parameter Models')
		print('	ALL:')
		model_parameters_ALL = perform_grid_search(MODEL,X_ALL,y_ALL,num_folds)
		save_parameters(param_file,model_parameters_ALL)			

	return model_parameters_MIR, model_parameters_SPO, model_parameters_LYR, model_parameters_ALL

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

def save_npy(X,X_val,y,y_val,DataSet,pl_indx,classification_set):
	print('- saving Training/Validation Matrixes:')
	print(pl_indx)
	print(classification_set)
	print(DataSet)

	path__ = home+'/SETS/EXP1/'
	if not os.path.exists(path__):
		os.makedirs(path__)

	path__ = path__+'PL_'+str(pl_indx)+'/'
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

def save_clf(clf,clf_type,dataset,pl_indx):
	if not path.exists(home+'classifiers/EXP1/'):
		os.mkdir(home+'classifiers/EXP1/')

	if not path.exists(home+'classifiers/EXP1/'+'PL_SET_'+str(pl_indx)+'/'):
		os.mkdir(home+'/classifiers/EXP1/'+'PL_SET_'+str(pl_indx)+'/')	

	clf_dir = home+'classifiers/EXP1/'+'PL_SET_'+str(pl_indx)+'/'+dataset+'/'

	if not path.exists(clf_dir):
		os.mkdir(clf_dir)

	clf_file = clf_dir+clf_type+'.pkl'

	with open(clf_file,"wb") as file:
		pickle.dump(clf, file, pickle.HIGHEST_PROTOCOL)


def save_parameters(param_file,params):
	
	with open(param_file,"wb") as file:
		pickle.dump(params, file, pickle.HIGHEST_PROTOCOL)

def create_ALL_df(MIR_df,Sp_df,Lyr_df):

	ALL_df = MIR_df.join([Sp_df,Lyr_df])
	
	return ALL_df

def do_pca(X):
	print("*********** SVD + PCA ***********")

	#print("SVD")
	value = np.linalg.svd(X,compute_uv=False)
	#print(value)
	#print("SORTED")
	value_s = sorted(value)
	#print(value_s)
	#print("INVERT")
	values_si = value_s[::-1]
	#print(values_si)

	sum_ = []
	

	i=0
	while i < len(values_si): 
		#print i
		if i == 0:
			sum_.append(values_si[i])
		
		else:
			sum_.append(sum_[i-1]+values_si[i])

		i += 1

	#print("SUM")
	#print(sum_)

	## gerade
	#gerade = range(int(sum_[0]),int(sum_[-1]),int(sum_[-1]-sum_[0])/len(sum_))
	
	## distance
	curve = sum_
	nPoints = len(curve)
	allCoord = np.vstack((range(nPoints), curve)).T
	np.array([range(nPoints), curve])
	firstPoint = allCoord[0]
	lineVec = allCoord[-1] - allCoord[0]
	lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
	vecFromFirst = allCoord - firstPoint
	scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
	vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
	vecToLine = vecFromFirst - vecFromFirstParallel
	distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
	#global idxOfBestPoint
	idxOfBestPoint = np.argmax(distToLine)
	print('- INDEX OF ELBOW:')
	print(idxOfBestPoint)

	markers_on = [idxOfBestPoint]
	'''
	plt.figure(1)
	plt.plot(range(0,len(sum_)),sum_,color='g',marker='o',markevery=markers_on)
	#plt.plot(range(0,len(gerade)),gerade,color='b')
	plt.xlabel('Feature')
	plt.ylabel('Sum Singular Values')
	plt.title('SVD')
	plt.legend(['Index of bend: %d' % idxOfBestPoint],loc='center right',)
	plt.grid()
	plt.savefig('Figures/SVD-pl-sum.pdf')
	'''

	pca = PCA(n_components=idxOfBestPoint, svd_solver= 'full') 	# Doc:n_components == 'mle' and svd_solver == 'full' Minka's MLE is used to guess the dimension 
	#print(X.shape)
	pca.fit(X)
	X_new = pca.transform(X)

	#print(X_new)
	#print(X_new.shape)
	
	
	return X_new
	
def open_parameters(param_file):
	
	file = open(param_file, 'rb')
	parameters = pickle.load(file)
	file.close()
	
	return parameters

def print_save(pl_indx,pl_list,data_set, clf, y_true, y_pred, X_val, y_val, params):

	print(" # * # * # * # * # * # * # * #")
	print(" - PLAYLIST TRAINED TO CLF")
	print(' PL CLF index:')
	print(" - "+str(pl_indx))
	print(" - - - - - - - - - - - - - - -")
	## unpack tuples
	print(" - TRAINED WITH:")
	for i,pl in enumerate(pl_list):
		print(' - PL ##'+str(i))
		print(pl)
	print(" -------- RESULTS --------")
	print(" ----- "+data_set+"-----")
	print(' -'+clf+' Classifier ')
	print(' - PARAMETERS:')
	print(params())
	print(' - size of evaluation set')
	print(X_val.shape)
	print(y_val.shape)
	print(' --> Validation: ')
	print(' - Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred))
	print(confusion_matrix(y_true,y_pred))
	print()
	print(" - - - - - - - - - - - ")
	print(" ")
	print(" # * # * # * # * # * # * # * #")

	exp_1_results = home+"CLF_results/EXP_1/"
	if not path.exists(exp_1_results):
		os.mkdir(exp_1_results)

	text_file = open(exp_1_results+data_set+" - "+clf+".txt", "a+")
	text_file.write(" # * # * # * # * # * # * # * #\n")
	text_file.write(" - PLAYLIST TRAINED TO CLF\n")
	text_file.write(' PL CLF index:')
	text_file.write(" - "+str(pl_indx))
	text_file.write(" - - - - - - - - - - - - - - -\n")
	for i, pl in enumerate(pl_list):
		text_file.write(' - ############:\n'+str(i)+"\n")
		text_file.write(' - USER:\n')
		text_file.write(str(pl[0])+"\n")
		text_file.write(' - PLAYLIST:')
		text_file.write(str(pl[1])+"\n")
		text_file.write(' - PL ID')
		text_file.write(str(pl[2])+"\n")
	text_file.write(" -------- RESULTS --------\n")
	text_file.write(" ----- "+data_set+"-----\n")
	text_file.write(' -'+clf+' Classifier \n')
	text_file.write(' - PARAMETERS:\n')
	param_str_= json.dumps(params())
	text_file.write(param_str_)
	#text_file.write(' - size of evaluation set\n')

	#X_val_shape = X_val.shape
	#y_val_shape = y_val.shape
	
	#x_shape = ''.join(X_val_shape)
	#y_shape = ''.join(y_val_shape)
	#text_file.write(x_shape)
	#text_file.write('\n')
	#text_file.write(x_shape)
	text_file.write('\n')
	text_file.write(' --> Validation: ')
	text_file.write('\n')
	text_file.write(' - Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	text_file.write('\n')
	text_file.write(classification_report(y_true, y_pred))
	text_file.write('\n')
	text_file.write(np.array2string(confusion_matrix(y_true,y_pred)))
	text_file.write('\n')
	text_file.write(" - - - - - - - - - - - ")
	text_file.write('\n')
	text_file.write(" ")
	text_file.write('\n')
	text_file.write(" # * # * # * # * # * # * # * #")
	text_file.write('\n')
	text_file.close()

def del_previous_results():

	parameters_folder = home+'parameters/EXP1/*/*/*.pkl'
	files = glob.glob(parameters_folder)
	for f in files:
		os.remove(f)

	#		sets exp1 PL_setID DS training-OR-validation  *.npy
	
	for file in clf_res:
		if file.startswith('EXP1'):
			print(file)
			os.remove(clf_res+file)

def get_acc(file):

	with open(clf_res+file,'r') as the_file:
		L = []
		for line in the_file:
			if line.startswith(' - Accuracy'):
				if len(L) < 9:
					L.append(line)
	return L 


def get_res(_clf_):
	
	results_ = clr_res+'EXP_1/'

	for file in os.listdir(results_):
		if file.endswith(_clf_+'.txt'):
			if file.startswith('MIR'):
				acc_L_MIR = get_acc(file)
			if file.startswith('SPO'):
				acc_L_LYR = get_acc(file)
			if file.startswith('LYR'):
				acc_L_SPO = get_acc(file)
			if file.startswith('ALL'):
				acc_L_ALL = get_acc(file)
			else:
				pass

	MIR_acc = [float(item[13:][:-2]) for item in acc_L_MIR]
	LYR_acc = [float(item[13:][:-2]) for item in acc_L_LYR]
	SPO_acc = [float(item[13:][:-2]) for item in acc_L_SPO]
	ALL_acc = [float(item[13:][:-2]) for item in acc_L_ALL]
	
	#print('----')
	#print(MIR_acc)
	#print(len(MIR_acc))

	plotter(_clf_,MIR_acc,SPO_acc,LYR_acc,ALL_acc)

def plotter(_clf_,MIR_acc,SPO_acc,LYR_acc,ALL_acc):
	x_ = [2,3,4,5,6,7,8,9,10]

	# x axis - amount of PL
	# y axis - classification accuracy 

	plt.figure()
	plt.ylim(35,102.5)
	plt.xlabel('Amount of playlists')
	plt.ylabel('Classification accuracy '+_clf_+' [%]')
	plt.plot(x_,MIR_acc,marker='.',color='red',label='MIR')
	plt.plot(x_,SPO_acc,marker='.',color='#1DB954',label='SPO')
	plt.plot(x_,LYR_acc,marker='.',color='blue',label='LYR')
	plt.plot(x_,ALL_acc,marker='.',color='black',label='ALL')
	plt.legend(loc=1)
	plt.savefig(home+'Figures/ex-1_'+_clf_+'.eps',format='eps')


init()