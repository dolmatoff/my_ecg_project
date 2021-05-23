% Filter parameters:
% d  : filter order parameter (the filter is of order 2d)
% fc : cut-off frequency (0 < fc < 0.5) (cycles/sample)

% SASS parameters
% K : order of sparse derivative
% lam : regularization parameter

% Uncomment one of the following lines to select parameter values
% PARAMVALS = 1;
PARAMVALS = 2;

switch PARAMVALS
    case 1
        % low-pass filter parameters
        d = 1;             % 2nd order filter
        fc = 0.03;
        % SASS parameters
        K = 2;
        lam = 1.3;
    case 2
        % low-pass filter parameters
        d = 2;             % 4th order filter
        fc = 0.03;
        % SASS parameters
        K = 2;
        lam = 1.52;
end


myFolder = 'C:\Users\Margo\source\repos\ecg-diagnosis\data\CPSC'
targetFolder = 'C:\Users\Margo\source\repos\ecg-diagnosis\data\CPSC\one_leads'
filePattern = fullfile(myFolder, '*.mat');
matFiles = dir(filePattern);

%% Define low-pass filter

d = 2;          % d : filter order parameter (d = 1, 2, or 3)
fc = 0.022;     % fc : cut-off frequency (cycles/sample) (0 < fc < 0.5);

%% SASS - run algorithm

K = 1;          % K : order of sparse derivative
lam = 1.2;      % lam : regularization parameter

compute_rmse = @(err) sqrt(mean(abs(err(:)).^2));

for k = 1:length(matFiles)
  baseFileName = matFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  all_leads = load(fullFileName);
  y = all_leads.val(1,:);
  y = normalize(y);
  y_col = y(:);
  
  N = length(y);	% N : length of signal
  n = (1:N)';
  ymax = max(y);
  ymin = min(y);
  
  [x, u] = sass_L1(y, d, fc, K, lam);
  % x : denoised signal
  % u : sparse order-K derivative
  %ax = [0 N ymin ymax]; 

  %figure(1)
  %clf
  %subplot(3, 1, 1)
  %plot(n, y_col)
  %title('Noisy ECG');
  %axis(ax)

  %subplot(3, 1, 2)
  %plot(n, x)
  %title('SASS')
  %axis(ax)
  
  save(fullfile(targetFolder, baseFileName), 'x')

end


