function [ feature_vect_no_str ] = features_ext_3( file )
p=cd;
addpath((strcat(p,'/MIRToolboxFULL')));
addpath((strcat(p,'/MIRToolboxFULL/AuditoryToolbox')));
%disp(file)
% This function will calculate all features of a given song 
tic 
[y, sr] = audioread(file);
disp('Time to load song')
toc

[m,n] = size(y);        % dimensions
song_size = [m,n];



%% STEREO vs MONO (we want the audio in Mono) 
% stereo:
if n == 2
    y_left  = y(:,2);
    y_right = y(:,1);

    y_mono  = (y_left + y_right)/2;
    %y_mono2 = (y_left + y_right)/2;
    disp('Song converted to mono')
    
% mono:
else
    y_mono = y;
    disp('Song was already mono')

end

%% Only choose 30s of the Audio. This is the preset when downloading the Spotify Tracks, but a local (genre) library should be cut to 30s for fairness

% song longer than 30s:
if m > 1399100 %1343238 % = 30.4589s    [ 1324153 = 30.0261]
    disp('Song is longer than 30s')
    
    % trim start
    trim = 1323000;
    y_mono(1:trim) = [];
    
    % trim end
    y_mono(1:end-trim) = [];
    %disp(y_mono)
    % one_liner
    %y_mono = y_mono(trim+1 , end-trim)
    
    % result: song -30s at the beginning and -30s at the end 
    
    [m2,n_mono] = size(y_mono);
    song_size = [m2,n_mono];
    
    upper_limit = m2%-1350000;
    clearvars random_int
    random_int = randi([1 upper_limit],1,1);

    y_mono_def = y_mono(random_int : random_int + 1323000);
    [m3,n_mono] = size(y_mono_def);
    song_size = [m3,n_mono];
    
    y_mono = y_mono_def;

%% Debugging:
elseif m < 1320000
    disp('Error Song too short')
    varargout{1} = NaN;       
    varargout{2} = NaN;        
    varargout{3} = NaN;        
    varargout{4} = NaN;        
    varargout{5} = NaN;        
    varargout{6} = NaN;
    varargout{7} = NaN;
    varargout{8} = NaN;
    varargout{9} = NaN;
    varargout{10}= NaN;
    varargout{11}= NaN;
    varargout{12}= NaN;
    varargout{13}= NaN;
    varargout{14}= NaN;
    varargout{15}= NaN;
    varargout{16}= NaN;
    varargout{17}= NaN;
    varargout{18}= NaN;
    varargout{19}= NaN;
    varargout{20}= NaN;
    varargout{21}= NaN;
    varargout{22}= NaN;
    varargout{23}= NaN;
    varargout{24}= NaN;
    varargout{25}= NaN;
    varargout{26}= NaN;
    varargout{27}= NaN;
    varargout{28}= NaN;
    varargout{29}= NaN;
    varargout{30}= NaN;
    varargout{31}= NaN;
    varargout{32}= NaN;
    varargout{33}= NaN;
    varargout{34}= NaN;
    varargout{35}= NaN;
    varargout{36}= NaN;
end   
 

    %disp('here')
    
    %sound(y_mono,sr)

    x       = miraudio(y_mono);


    filters_10  = mirfilterbank(x,'Gammatone');%'NbChannels',10); % cite paper why

    filtered_frames = mirframe(filters_10);

    filtered_spectrum = mirspectrum(filtered_frames);
    
    frames      = mirframe(x);
    
    spectrum    = mirspectrum(frames);


    spect_centroid      = mircentroid(spectrum);

    rolloff_85          = mirrolloff(frames);

    brightness          = mirbrightness(spectrum);

    spect_flux          = mirflux(spectrum);

    
    %% with the temporal domain 
    zerocross           = mirzerocross(x);

    RMS                 = mirrms(frames);

    lowenergy           = mirlowenergy(RMS);

    ASR                 = mirlowenergy(RMS, 'ASR');

    tempo               = mirtempo(x); 

    pulseclarity        = mirpulseclarity(x);

    onsets              = mironsets(x);

    key                 = mirkey(x);

    key_mode            = mirmode(x);

    % ver 3 added features 

    mfcc                = mirmfcc(spectrum, 'Bands', 10);

    pitch               = mirpitch(x, 'Autocor','Gammatone');%,'Total',5)

    event_dens          = mireventdensity(x);

    roughness           = mirroughness(spectrum);

    cepstrum            = mircepstrum(x);

    envelope            = mirenvelope(x);

    chroma              = mirchromagram(spectrum);

    key_st              = mirkeystrength(chroma);

    key_mode            = mirmode(key_st);

    ton_centroid        = mirtonalcentroid(chroma);






    %% EXPORT

    spect_centroid_ = exportation(spect_centroid);
    rolloff_85_     = exportation(rolloff_85);
    brightness_     = exportation(brightness);
    spect_flux_     = exportation(spect_flux);
    zerocross_      = exportation(zerocross);
    RMS_            = exportation(RMS);
    lowenergy_      = exportation(lowenergy);
    tempo_          = exportation(tempo);
    pulseclarity_   = exportation(pulseclarity);
    onsets_         = exportation(onsets);
    key_            = exportation(key);
    mode_           = exportation(key_mode);
    ASR_            = exportation(ASR);

    mfcc_           = exportation(mfcc);
    try
        pitch_         = exportation(pitch);
    catch 
        warning('Problem Calculating Pitch, assigning value NaN')
        pitch_ = 'NaN';
    end
    event_dens_     = exportation(event_dens);
    roughness_      = exportation(roughness);
    cepstrum_       = exportation(cepstrum);
    envelope_       = exportation(envelope);
    chroma_         = exportation(chroma);
    key_st_         = exportation(key_st);
    key_mode_       = exportation(key_mode);
    ton_centroid_   = exportation(ton_centroid);


    %% Values // Mean+STD
    
    spect_centroid_v    = double(spect_centroid_{2,1})

    rolloff85_mean      = double(rolloff_85_{2,1})
    rolloff85_std       = double(rolloff_85_{2,2})

    brightness_mean     = double( brightness_{2,1})
    brightness_std      = double(brightness_{2,2})

    spectralflux_mean   = double(spect_flux_{2,1})
    spectralflux_std    = double(spect_flux_{2,2})

    zerocross_rate      = double(zerocross_{2,1})

    RMS_energy_mean     = double(RMS_{2,1})
    RMS_energy_std      = double(RMS_{2,2})

    lowenergy_v         = double(lowenergy_{2,1})

    tempo_v             = double(tempo_{2,1})

    pulseclarity_v      = double(pulseclarity_{2,1})

    onsets_mean         = double(onsets_{2,1})
    onsets_std          = double(onsets_{2,2})
    onsets_peakpos      = double(onsets_{2,3})
    onsets_peakval      = double(onsets_{2,4})

    key_v               = double(key_{2,1})

    ASR_v               = double(ASR_{2,1})


    % MFCC
    mfcc_mean      = double(mfcc_{2,1})
    mfcc_std       = double(mfcc_{2,2})

    % Pitch 
    if ischar(pitch_) == 0  
        i = 1;
        len = size(pitch_);
        pitch_vect = [];
        while i <= len(2)
            pitch_vect= [pitch_vect, pitch_{2,i}];
            i=i+1;
        end
        pitch_mean = double(mean(pitch_vect))
        pitch_std  = double(std(pitch_vect))
    else 
        pitch_mean = NaN     % NaN
        pitch_std  = NaN     % NaN
    end
    % Event Density
    event_dens_v        = double(event_dens_{2,1})

    % Roughness 
    roughness_mean     = double(roughness_{2,1})
    roughness_std      = double(roughness_{2,2})

    % Cepstrum 
    cepstrum_mean     = double(cepstrum_{2,1})
    cepstrum_std      = double(cepstrum_{2,2})

    % Envelope
    envelope_mean     = double(envelope_{2,1})
    envelope_std      = double(envelope_{2,2})

    % Chroma
    chroma_mean     = double(chroma_{2,1})
    chroma_std      = double(chroma_{2,2})

    % Key Strength
    key_st_mean     = double(key_st_{2,1})
    key_st_std      = double(key_st_{2,2})

    % Key Mode
    key_mode_v        = double(key_mode_{2,1})

    % Tonal Centroid
    ton_centroid_mean     = double(ton_centroid_{2,1})
    ton_centroid_std      = double(ton_centroid_{2,2})

    %disp(rolloff_85_)
    %disp(brightness_)



    feature_vect_no_str = [spect_centroid_v ,rolloff85_mean, rolloff85_std,brightness_mean,brightness_std,spectralflux_mean,spectralflux_std,zerocross_rate,RMS_energy_mean,RMS_energy_std,lowenergy_v,tempo_v,pulseclarity_v,onsets_mean,onsets_std,onsets_peakpos,onsets_peakval,key_v, ASR_v, mfcc_mean , mfcc_std ,pitch_mean, pitch_std,event_dens_v, roughness_mean,roughness_std, cepstrum_mean,cepstrum_std,chroma_mean,chroma_std, key_st_mean,key_st_std,key_mode_v,ton_centroid_mean,ton_centroid_std];

end

    
    %feature_vect =  num2str([spect_centroid_v ,rolloff85_mean, rolloff85_std,brightness_mean,brightness_std,spectralflux_mean,spectralflux_std,zerocross_rate,RMS_energy_mean,RMS_energy_std,lowenergy_v,tempo_v,pulseclarity_v,onsets_mean,onsets_std,onsets_peakpos,onsets_peakval,key_v ,key_mode_v , ASR_v, mfcc_mean , mfcc_std ,pitch_mean, pitch_std,event_dens_v, roughness_mean,roughness_std, cepstrum_mean,cepstrum_std,chroma_mean,chroma_std, key_st_mean,key_st_std,ton_centroid_mean,ton_centroid_std])

    %feature_vector =        [ num2str(spect_centroid_v) ,num2str(rolloff85_mean), num2str(rolloff85_std),num2str(brightness_mean),num2str(brightness_std),num2str(spectralflux_mean),num2str(spectralflux_std),num2str(zerocross_rate),num2str(RMS_energy_mean),num2str(RMS_energy_std),num2str(lowenergy_v),num2str(tempo_v),num2str(pulseclarity_v),num2str(onsets_mean),num2str(onsets_std),num2str(onsets_peakpos),num2str(onsets_peakval),num2str(double(key_v )),num2str(key_mode_v ), num2str(ASR_v), num2str(mfcc_mean), num2str(mfcc_std) ,num2str(pitch_mean), num2str(double(pitch_std)),num2str(event_dens_v), num2str(roughness_mean),num2str(roughness_std), num2str(cepstrum_mean),num2str(cepstrum_std),num2str(chroma_mean),num2str(chroma_std), num2str(key_st_mean),num2str(key_st_std),num2str(key_mode_v),num2str(ton_centroid_mean),num2str(ton_centroid_std) ]
    %feature_vector = [ double(spect_centroid_v) ,double(rolloff85_mean), double(rolloff85_std),double(brightness_mean),double(brightness_std),double(spectralflux_mean),double(spectralflux_std),double(zerocross_rate),double(RMS_energy_mean),double(RMS_energy_std),double(lowenergy_v),double(tempo_v),double(pulseclarity_v),double(onsets_mean),double(onsets_std),double(onsets_peakpos),double(onsets_peakval),double(key_v ),double(key_mode_v ), double(ASR_v), double(mfcc_mean), double(mfcc_std) ,double(pitch_mean), double(pitch_std),double(event_dens_v), double(roughness_mean),double(roughness_std), double(cepstrum_mean),double(cepstrum_std),double(chroma_mean),double(chroma_std), double(key_st_mean),double(key_st_std),double(key_mode_v),double(ton_centroid_mean),double(ton_centroid_std) ]
    %feature_vector = [ strings(spect_centroid_v) ,strings(rolloff85_mean), strings(rolloff85_std),strings(brightness_mean),strings(brightness_std),strings(spectralflux_mean),strings(spectralflux_std),strings(zerocross_rate),strings(RMS_energy_mean),strings(RMS_energy_std),strings(lowenergy_v),strings(tempo_v),strings(pulseclarity_v),strings(onsets_mean),strings(onsets_std),strings(onsets_peakpos),strings(onsets_peakval),strings(key_v ),strings(key_mode_v ), strings(ASR_v), strings(mfcc_mean), strings(mfcc_std) ,strings(pitch_mean), strings(pitch_std),strings(event_dens_v), strings(roughness_mean),strings(roughness_std), strings(cepstrum_mean),strings(cepstrum_std),strings(chroma_mean),strings(chroma_std), strings(key_st_mean),strings(key_st_std),strings(key_mode_v),strings(ton_centroid_mean),strings(ton_centroid_std) ]
    %global feature_vector
    %disp(feature_vect)
    %class(feature_vect)
    %feature_vect = ({spect_centroid_v ;rolloff85_mean; rolloff85_std;brightness_mean;brightness_std;spectralflux_mean;spectralflux_std;zerocross_rate;RMS_energy_mean;RMS_energy_std;lowenergy_v;tempo_v;pulseclarity_v;onsets_mean;onsets_std;onsets_peakpos;onsets_peakval;key_v ;key_mode_v ; ASR_v; mfcc_mean ; mfcc_std ;pitch_mean; pitch_std;event_dens_v; roughness_mean;roughness_std;cepstrum_mean;cepstrum_std;chroma_mean;chroma_std; key_st_mean;key_st_std;key_mode_v;ton_centroid_mean;ton_centroid_std})
    
    %varargout{1} = spect_centroid_v ;       %NaN problem
    %varargout{2} = rolloff85_mean;          %NaN problem
    %varargout{3} = rolloff85_std;           %NaN problem
%     varargout{4} = brightness_mean;         %NaN problem
%     varargout{5} =brightness_std;           %NaN problem
%     varargout{6} =spectralflux_mean;
%     varargout{7} =spectralflux_std;
%     varargout{8} =zerocross_rate;
%     varargout{9} =RMS_energy_mean;
%     varargout{10}=RMS_energy_std;
%     varargout{11}=lowenergy_v;
%     varargout{12}=tempo_v;
%     varargout{13}=pulseclarity_v;
%     varargout{14}=onsets_mean;
%     varargout{15}=onsets_std;
%     varargout{16}=onsets_peakpos;
%     varargout{17}=onsets_peakval;
%     varargout{18}=key_v;
%     varargout{19}=key_mode_v;
%     varargout{20}=ASR_v;
%     varargout{21}=mfcc_mean;
%     varargout{22}=mfcc_std;
%     varargout{23}=pitch_mean;
%     varargout{24}=pitch_std;
%     varargout{25}=event_dens_v;
%     varargout{26}=roughness_mean;
%     varargout{27}=roughness_std;
%     varargout{28}=cepstrum_mean;
%     varargout{29}=cepstrum_std;
%     varargout{30}=chroma_mean;
%     varargout{31}=chroma_std;
%     varargout{32}=key_st_mean;
%     varargout{33}=key_st_std;
%     varargout{34}=key_mode_v;
%     varargout{35}=ton_centroid_mean;
%     varargout{36}=ton_centroid_std;
%     length(varargout)
    %;;;;;;;;;;;;;; ; ; ;  ;  ;; ;; ;;;;;; ;;;;
%end



% ALL FEATURES EXTRACTION FROM MONO:
%       1.- Dynamics
%       2.- Fluctuation
%       3.- Rhythm 
%       4.- Spectral
%       5.- Timbre
%       6.- Tonal
   

%feat    = mirfeatures(x)
   
   
% Make them all global variables 
%global RMS_energy tempo attack_time attack_slope centroid brightness spread skewness kurtosis rolloff95 rolloff85 spectentropy flatness roughness irregularity zerocross lowenergy spectralflux  chromagram key key_mode keyclarity mode HCDF
%global RMS_energy_mean RMS_energy_std tempo_value attack_time_mean attack_time_std attack_slope_mean attack_slope_std centroid_value brightness_mean brightness_std spread_value skewness_value kurtosis_value rolloff95_mean rolloff95_std rolloff85_mean rolloff85_std spectentropy_mean spectentropy_std flatness_mean flatness_std
%global roughness_mean roughness_std irregularity_mean irregularity_std zerocross_mean zerocross_std lowenergy_value spectralflux_mean spectralflux_std chromagram_value key_value key_mode_value keyclarity_mean keyclarity_std mode_mean mode_std HCDF_mean HCDF_std
%global ASR 
% 1.- Dynamics extraction

%RMS_energy  = exportation(feat.dynamics.rms)                        % mean + std 2x2 cell array


% 2.- Fluctuation

% NADA

% 3.- Rhythm

%tempo           = exportation(feat.rhythm.tempo)                    % 2x1 cell array
%attack_time     = exportation(feat.rhythm.attack.time)              % mean + std 2x2 cell array
%attack_slope    = exportation(feat.rhythm.attack.slope)             % mean + std 2x2 cell array


% 4.- Spectral 

%centroid        = exportation(feat.spectral.centroid)               % 2x1 cell array                        <<<<<<
%brightness      = exportation(feat.spectral.brightness)             % mean + std 2x2 cell array
%spread          = exportation(feat.spectral.spread)                 % 2x1 cell array
%skewness        = exportation(feat.spectral.skewness)               % 2x1 cell array
%kurtosis        = exportation(feat.spectral.kurtosis)               % 2x1 cell array
%rolloff95       = exportation(feat.spectral.rolloff95)              % mean + std 2x2 cell array             
%rolloff85       = exportation(feat.spectral.rolloff85)              % mean + std 2x2 cell array             <<<<<<
%spectentropy    = exportation(feat.spectral.spectentropy)           % mean + std 2x2 cell array
%flatness        = exportation(feat.spectral.flatness)               % mean + std 2x2 cell array
%roughness       = exportation(feat.spectral.roughness)              % mean + std 2x2 cell array
%irregularity    = exportation(feat.spectral.irregularity)           % mean + std 2x2 cell array
% mfcc // dmfcc // ddmfcc faltan no desde mirtoolbox

%Mel             = mirspectrum(x,'Mel');
%magnitude       = get(mel,'Magnitude');
%magnitude       = magnitude{1}{1}
%mag_mean        = mean(magnitude)
%mag_var         = var(magnitude)



% 5.- Timbre

%zerocross       = exportation(feat.timbre.zerocross)                % mean + std 2x2 cell array             <<<<<<
%lowenergy       = exportation(feat.timbre.lowenergy)                % 2x1 cell array
%spectralflux    = exportation(feat.timbre.spectralflux)             % mean + std 2x2 cell array             <<<<<<


% 6.- Tonal 

%spectrum        = mirspectrum(x);
%chromagram_     = mirchromagram(spectrum);
%chromagram      = exportation(feat.tonal.chromagram.centroid)       % 2x1 cell array
%keystrength     = mirkeystrength(chromagram_);
%ton_peaks       = mirpeaks(keystrength);
%key_            = mirkey(ton_peaks);
%key             = exportation(key_)                                 % 2x1 cell array
%key_mode_       = mirmode(keystrength);
%key_mode        = exportation(key_mode_)                            % 2x1 cell array

%keyclarity      = exportation(feat.tonal.keyclarity)                % mean + std 2x2 cell array
%mode            = exportation(feat.tonal.mode)                      % mean + std 2x2 cell array
%HCDF            = exportation(feat.tonal.hcdf)                      % mean + std 2x2 cell array


% 7.- Mood-Detection

%ASR             = exportation(mirlowenergy(x, 'ASR'))
%mASR            = mean(ASR)
%stdASR          = std(ASR)

%% Exportation out of Cell-Array
