function [] = FinalProject()

dbstop if error; %halts the script without stopping it when an error occurs, good for debugging

phone = 'aa';

if exist(strcat('nxt_',phone,'.mat'),'file')
    fprintf('loading phones info...');
    load(strcat('nxt_',phone,'.mat'));
    fprintf('done\n');
else
    find_phone(phone);
end

if exist(strcat(phone,'.mat'),'file')
    runSVM(phone);
    
else
    analyze_person(1,phone);
    analyze_person(2,phone);
end

end

function [] = runSVM(phone)
%manually saved this file
load(strcat(phone,'.mat'));

%load F0s
F0_1 = P1.f0;
F0_2 = P2.f0;

%load h1h2
H_1 = P1.h1h2;
H_2 = P2.h1h2;

% %Select elements to be part of the training set (randomly sample a random
% %amount)
% train1 = logical(round(rand(length(H_1),1)));
% train2 = logical(round(rand(length(H_2),1)));
%
% if sum(train1)>(length(H_1)/2)
%     train1 = ~train1;
% end
% if sum(train2)>(length(H_2)/2)
%     train2 = ~train2;
% end
%
% %make training set
% H_1Tr = H_1(train1);
% F0_1Tr = F0_1(train1);
% H_2Tr = H_2(train2);
% F0_2Tr = F0_2(train2);
%
% %make test set
% H_1Te = H_1(~train1);
% F0_1Te = F0_1(~train1);
% H_2Te = H_2(~train2);
% F0_2Te = F0_2(~train2);

%make training set
%400 came from the find_test_set_size
k = min(400,min(length(H_1),length(H_2)));
[H_1Tr,i1] = datasample(H_1,k);
F0_1Tr = F0_1(i1);
[H_2Tr,i2] = datasample(H_2,k);
F0_2Tr = F0_2(i2);

%indices for test set
x1 = ones(length(H_1),1);
x1(i1) = 0;
x1 = logical(x1);
x2 = ones(length(H_2),1);
x2(i2) = 0;
x2 = logical(x2);

%make test set
H_1Te = H_1(x1);
F0_1Te = F0_1(x1);
H_2Te = H_2(x2);
F0_2Te = F0_2(x2);

Training = [H_1Tr , F0_1Tr; H_2Tr , F0_2Tr];
GroupTrain = [ones(length(H_1Tr),1);zeros(length(H_2Tr),1)];

%GET SVM
svm = svmtrain(Training,GroupTrain,'kernel_function','rbf','method','LS');

%check accuracy
G = svmclassify(svm,Training);
correct = sum(G==GroupTrain);
accuracy = correct/length(GroupTrain);

Testing = [H_1Te , F0_1Te; H_2Te , F0_2Te];
%actual results
GroupTest = [ones(length(H_1Te),1);zeros(length(H_2Te),1)];

%use classifier
GroupSVM = svmclassify(svm,Testing);

%Check accuracy
correctTest = sum(GroupTest==GroupSVM);
accuracyTest = correctTest/length(GroupSVM);

save(strcat(phone,'.mat'),'svm','Training','GroupTrain','Testing','GroupTest','accuracy','-append');

end

function [] = analyze_person(i,phone)
swbdir = '/Volumes/My Passport/CompLingCorpus'; %directory where my corpus is

swbfiles = rdir( strcat(swbdir,'/**/*.sph'));

if isempty(swbfiles), fprintf('Whoops! No .sph files found\n'); return; end

if exist(strcat('nxt_',phone,'.mat'),'file')
    fprintf('loading phones info...');
    load(strcat('nxt_',phone,'.mat'));
    fprintf('done\n');
end

% %graphic_setup
% display_rng = [-2 2];      %(s) time range for display
% fontsize    = 16;          %change this to adjust fontsizes
% fontname    = 'Verdana';   %change
% [btn,tb] = graphics_setup(fontname,fontsize);

%calculate fourier transform for a group of males (fft)
Person = Males{i};

%find the portion of the .sph file we are looking for
swbfileid = Person.conversationID;

%here the filename is found from our list of swb .sph files:
cellwithstringmatches = strfind({swbfiles.name},swbfileid);
sphfilename = swbfiles(~cellfun('isempty',cellwithstringmatches));

if isempty(which('readsph')), fprintf('Whoa--is the voicebox toolbox on your path?'); return; end

%use readsph from the voicebox toolbox (doc readsph) to read the
%sphere formant audio file (use the 'p' flag):

[wavboth,Fs] = readsph(sphfilename.name,'p',-1,0);

%the audio samples (the variable 'wavboth') has two columns (in this case channels), one for each
%of the speakers in the conversation. column 1 corresponds to speaker A
%and column 2 to speaker B.

swbchanid = Person.Gender;

%figure out whether the tokens are from speaker A or B

if swbchanid == 'A',
    wav = wavboth(1:length(wavboth));
else
    wav = wavboth(length(wavboth)+1:end);
end

%here is a vector of times for the waveform:
t = linspace(0,length(wav)-1,length(wav))/Fs;

numberOfTokens = length(Person.t0);
f0 = zeros(numberOfTokens,1);
h1h2 = zeros(numberOfTokens,1);

validPhone = zeros(numberOfTokens,1);

fprintf('analyzing tokens of %s. %d tokens total.\n',Person.SpeakerID,numberOfTokens);
for j=1:numberOfTokens
    fprintf('analyzing token %d out of %d\n', j,numberOfTokens);
    
    cutOff = .01;
    
    %use new start and end values that eliminate noise on the sides
    startVal = Person.t0(j) + cutOff;
    endVal = Person.t1(j) - cutOff;
    
    tStart =  startVal*Fs + 1;
    tEnd = endVal*Fs + 1;
    
    wavPhone = wav(round(tStart):round(tEnd));
    tAdjusted = t(round(tStart):round(tEnd));
    
    %calculate fourier transform
    fourier = fft(wavPhone);
    
    %do 10*log_10(|fourier|) to get it on a decibel
    fourier = 10*log10(abs(fourier));
    
    %smooth the fourier transform
    %fourierSmooth = smooth(fourier);
    
    %FIND F0
    [amp,locs] = findpeaks(fourier);
    
    %throw out ones whose amplitudes are low
    %take all amplitudes and z-score them
    %exclude any peaks that are below some threshold in z.  The threshold
    %would be fairly high
    
    ampZ = zscore(amp);
    
    thresh = 1;
    
    logicalZ = (ampZ > thresh);
    
    newAmp = amp(logicalZ);
    newLoc = locs(logicalZ);
    
    %take two remaining peaks with lowest frequency
    %sanity check: distance between first two peaks is between 75 - 500 Hz
    %f0 = distance between first two peaks.
    
    if numel(newLoc)>=2
        f0(j) = newLoc(2) - newLoc(1);
        if f0(j)>=75 || f0(j)<=500
            validPhone(j) = 1;
        end
    end
    
    %FIND H1 - H2
    %amplitude of first peak - amplitude of second peak
    if numel(newAmp)>=2
        h1h2(j) = newAmp(1) - newAmp(2);
    end
    
    %     %plot the waveform as a function of time
    %     plot(t,wav,'k');
    %
    %     %some stuff you can ignore:
    %     hold on;
    %     set(gca,'Position',[0.05 0.05 0.725 0.90]);
    %     token_rng = [startVal endVal];
    %     wavix = round(token_rng*Fs);
    %     wavix(wavix<1) = 1;
    %     wavix(wavix>length(wav)) = length(wav);
    %     xlim(token_rng);
    %     drawnow;
    %
    %     %setup audio playback and play
    %     playerobj = audioplayer(wav(wavix(1):wavix(2)),Fs);  %>>doc audioplayer, >>doc play
    %     play(playerobj);
    
end

fprintf('done\n');

validPhone = logical(validPhone);

%eliminate ones whose f0 values are not in the range
f0 = f0(validPhone);
h1h2 = h1h2(validPhone);

Person.f0 = f0;
Person.h1h2 = h1h2;
Person.t0 = Person.t0(validPhone);
Person.t1 = Person.t1(validPhone);

fprintf('saving...');
save(strcat(Person.SpeakerID,'_',phone,'.mat'),'Person');
fprintf('done\n');
end

function [] = find_phone(phone)
%to find the phone in the corpus

fprintf('finding phones...');

if exist('nxt_phonx.mat','file') && exist('nxt_resources.mat','file')
    load('nxt_phonx.mat');
    load('nxt_resources.mat');
    
    %indices of the phone in phnx
    i = strfind(phnx,phone);
    
    %speaker details
    spkrIndex = phn_spkr(i);
    speakers = SPKR(spkrIndex);
    uniqueSpeakers = unique(speakers);
    
    %make cell array of structs. Space for A and B for each conversation
    uniquePhone = cell(length(uniqueSpeakers),1);
    
    %SEPARATE TOKENS BY SPEAKER
    for i=1:length(uniqueSpeakers)
        %index of the unique speaker in SPKR
        speakerID = uniqueSpeakers{i};
        conversationID = speakerID(1:end-1);
        speakerIdentifier = speakerID(end);
        
        indOfSpeaker = find(strcmp(SPKR, speakerID));
        
        logicalSpeaker = (phn_spkr==indOfSpeaker);
        
        %keep only primary stress
        logicalStress = phn_stress == 3;
        logicalSpeaker = logicalSpeaker(1:length(phn_stress)) & logicalStress;
        
        indicesOfPhones = strfind(logicalSpeaker,1);
        
        stress = phn_stress(logicalSpeaker(1:length(phn_stress)));
        
        t0 = phn_t0(logicalSpeaker);
        t1 = phn_t1(logicalSpeaker);
        
        %keep the duration of the phone a reasonable length
        %found the median duration for this person and selected ones
        %that were around it
        durations = round((t1-t0)*100)/100;
        med = mean(durations);
        
        margin = .03;
        
        indOfValid = find(durations>=(med-margin)&durations<=(med+margin));
        
        %remove ones that don't fit in the valid range for durations
        indicesOfPhones = indicesOfPhones(indOfValid);
        stress = stress(indOfValid);
        t0 = t0(indOfValid);
        t1 = t1(indOfValid);
        
        for j=1:length(DLG)
            if strcmp(DLG(j).swbid,conversationID)
                S = '';
                if speakerIdentifier=='A'
                    S = DLG(j).spkrA;
                else
                    S = DLG(j).spkrB;
                end
                
                for k=1:length(SPKR_Resources)
                    if(strcmp(SPKR_Resources(k).id,S))
                        Sstruct = SPKR_Resources(k);
                        gender = Sstruct.sex;
                        s = struct('conversationID', conversationID,'SpeakerID', S,'Gender', gender,'mean', med,'t0',t0,'t1',t1);
                        uniquePhone{i} = s;
                        break;
                    end
                end
                break;
            end
        end
        fprintf('Iteration: %d/%d\n',i,length(uniqueSpeakers));
    end
    
    isMale = zeros(size(uniquePhone));
    for i=1:length(uniquePhone)
        if(uniquePhone{i}.Gender=='M')
            isMale(i) = 1;
        end
    end
    
    isMale = logical(isMale);
    
    %Separate by gender
    Males = transpose(uniquePhone(isMale));
    Females = transpose(uniquePhone(~isMale));
    filename = strcat('nxt_',phone,'.mat');
    
    save(filename,'Males','Females');
    
else
    fprintf('Whoops! nxt_phonx.mat or nxt_resources.mat not found'); return;
end

fprintf('done\n');
end

function max = find_max(Group)

size = zeros(length(Group),1);

max = 0;

for i=1:length(Group)
    size(i) = length(Group{i}.t0);
    if (length(Group{i}.t0)>max)
        max = length(Group{i}.t0);
    end
end

end

%finds all the durations of the phone for the group to find a good median
function durations = find_durations(Group)

numberOfPhones = 0;

for i=1:length(Group)
    numberOfPhones = numberOfPhones + length(Group{i}.t0);
end

durations = zeros(numberOfPhones,1);

count = 1;
for i=1:length(Group)
    person = Group{1};
    for j=1:length(person.t0)
        durations(count) = person.t1(j) - person.t0(j);
        count = count + 1;
    end
end

durations = round(durations*100)/100;

%need to pick a good range
numberOfValidPhones = histc(durations,.03:.15);

end

% %you can ignore the functions below:
% function [btn,tb] = graphics_setup(fontname,fontsize)
% set(0,'defaultaxesfontname',fontname);
% set(0,'defaultuicontrolfontname',fontname);
% set(0,'defaulttextfontname',fontname);
% set(0,'defaultaxesfontsize',fontsize-6);
% set(0,'defaultuicontrolfontsize',fontsize);
% set(0,'defaulttextfontsize',fontsize-4);
% figure;
% set(gcf,'units','normalized','outerposition',[0 .1 1 .5]);
%
% btn.prev = uicontrol('Style','pushbutton','units','normalized','position',[.900 .80 .025 .10],...
%     'string','<','Callback','uiresume','fontsize',fontsize);
% btn.next = uicontrol('Style','pushbutton','units','normalized','position',[.925 .80 .025 .10],...
%     'string','>','Callback','uiresume','fontsize',fontsize);
% btn.listen = uicontrol('Style','pushbutton','units','normalized','position',[.80 .80 .09 .10],...
%     'string','listen','Callback','uiresume','fontsize',fontsize);
% btn.quit = uicontrol('Style','pushbutton','units','normalized','position',[.80 .70 .09 .10],...
%     'string','quit','Callback','uiresume','fontsize',fontsize);
% btn.skip = uicontrol('Style','pushbutton','units','normalized','position',[.925 .70 .025 .10],...
%     'string','>>','Callback','uiresume','fontsize',fontsize-2,'tooltipstr','Skip to next unclassified token');
% btn.first = uicontrol('Style','pushbutton','units','normalized','position',[.90 .70 .025 .10],...
%     'string','<<','Callback','uiresume','fontsize',fontsize-2,'tooltipstr','Back to beginning');
% tb.info = annotation('textbox','position',[.80 .90 .20 .10],'edgecolor','none','Fontsize',fontsize,'fontname',fontname);
% tb.index = annotation('textbox','position',[.90 .60 .075 .10],'edgecolor','none','Fontsize',fontsize,'fontname',fontname);
% end
%
% function [] = update_display(WRD,i,numwrd,tb)
% CNTX = WRD.context;
% colors = [1 0 0; 0 1 0];
% for j=1:length(CNTX)
%     tix = [CNTX(j).start CNTX(j).end];
%     if tix(1)<min(xlim) || tix(2)>max(xlim), continue; end
%     fill([tix(1) tix(2) tix(2) tix(1)],...
%         [min(ylim) min(ylim) max(ylim) max(ylim)],colors(mod(j,2)+1,:),'FaceAlpha',0.25);
%     text(tix(1)+0.005*diff(xlim),max(ylim)-(0.05*(mod(j,2)+1))*diff(ylim),CNTX(j).orth);
% end
%
% set(gca,'ButtonDownFcn','uiresume');
%
% set(tb.info,'string',WRD.msstate);
% set(tb.index,'string',[num2str(i,'%03.0f') '/' num2str(numwrd,'%03.0f')]);
% end



