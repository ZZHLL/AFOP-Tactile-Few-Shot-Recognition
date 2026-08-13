function [featurepool, output_class, meta] = build_feature_pool(dataset3, fs, num, featureConfig)
% Build the PVDF/SG feature pool from 4-by-4000 tactile trials.

dataset1   = dataset3;
if nargin<4 || isempty(featureConfig), featureConfig=struct(); end
if isfield(featureConfig,'wavelet'), wavf=char(featureConfig.wavelet); else, wavf='rbio2.2'; end
if isfield(featureConfig,'levels'), N_wavlayer=featureConfig.levels; else, N_wavlayer=3; end
Num_point  = size(dataset3{find(~cellfun(@isempty,dataset3),1)},2);
assert(Num_point == 4000, 'Expected 4-s, 4000-sample trials.');
assert(isscalar(N_wavlayer) && N_wavlayer==round(N_wavlayer) && N_wavlayer>=1, ...
    'featureConfig.levels must be a positive integer.');
assert(N_wavlayer<=wmaxlev(Num_point,wavf), ...
    'Wavelet %s supports at most %d levels for %d samples.', ...
    wavf,wmaxlev(Num_point,wavf),Num_point);

num_fea_cloumn = 58;
num_fea_row    = N_wavlayer*2;
num.feature    = num_fea_row*num_fea_cloumn + num_fea_row*2 + 13*2;
featurepool    = zeros(num.trial*num.etr*num.shape, num.feature);

readme = {'featurepool'; ...
    'Each level contributes 58 PVDF1 and 58 PVDF2 descriptors.'; ...
    'Each PVDF block contains 14 approximation-time, 14 detail-time, 15 approximation-frequency, and 15 detail-frequency descriptors.'; ...
    'The PVDF blocks are followed by normalized band-energy ratios and 13 time-domain descriptors for each SG channel.'};
Length = Num_point;
n_fft  = 2^nextpow2(Length);

Q         = 30;
harmonics = 150:50:450;
num.filter   = length(harmonics);
filter_ab    = cell(num.filter,2);
dataset_shifted = cell(size(dataset1));

for i = 1:num.shape
    for j = 1:num.etr*num.trial
        if isempty(dataset1{i,j}), continue; end 
        temp1 = dataset1{i,j}(1,:);
        temp2 = dataset1{i,j}(2,:);
        for k = 1:num.filter
            BW = harmonics(k)/Q;
            f0 = harmonics(k);
            [filter_ab{k,1},filter_ab{k,2}] = iirnotch(f0/(fs/2), BW/(fs/2));
            temp1 = filtfilt(filter_ab{k,1},filter_ab{k,2}, temp1);
            temp2 = filtfilt(filter_ab{k,1},filter_ab{k,2}, temp2);
        end
        dataset_shifted{i,j}(1,:) = temp1;
        dataset_shifted{i,j}(2,:) = temp2;
        dataset_shifted{i,j}(3,:) = dataset1{i,j}(3,:);
        dataset_shifted{i,j}(4,:) = dataset1{i,j}(4,:);
    end
end

for i = 1:num.shape
    for j = 1:num.trial
        for k = 1:num.etr
            if isempty(dataset_shifted{i,(j-1)*num.etr+k}), continue; end
            [row1,col1] = wavedec(dataset_shifted{i,(j-1)*num.etr+k}(1,:), N_wavlayer, wavf);
            [row2,col2] = wavedec(dataset_shifted{i,(j-1)*num.etr+k}(2,:), N_wavlayer, wavf);
            for m = 1:N_wavlayer
                sensordata.a1{i}{j,k}(m,:) = wrcoef('a', row1,col1,wavf,m);
                sensordata.d1{i}{j,k}(m,:) = wrcoef('d', row1,col1,wavf,m);
                sensordata.a2{i}{j,k}(m,:) = wrcoef('a', row2,col2,wavf,m);
                sensordata.d2{i}{j,k}(m,:) = wrcoef('d', row2,col2,wavf,m);

                sensordata.af1{i}{j,k}(m,:) = fft(sensordata.a1{i}{j,k}(m,:), n_fft);
                P2.af1 = abs(sensordata.af1{i}{j,k}(m,:)/Length);
                P1.af1 = P2.af1(1:Length/2+1);
                P1.af1(2:end-1) = 2*P1.af1(2:end-1);

                sensordata.df1{i}{j,k}(m,:) = fft(sensordata.d1{i}{j,k}(m,:), n_fft);
                P2.df1 = abs(sensordata.df1{i}{j,k}(m,:)/Length);
                P1.df1 = P2.df1(1:Length/2+1);
                P1.df1(2:end-1) = 2*P1.df1(2:end-1);

                sensordata.af2{i}{j,k}(m,:) = fft(sensordata.a2{i}{j,k}(m,:), n_fft);
                P2.af2 = abs(sensordata.af2{i}{j,k}(m,:)/Length);
                P1.af2 = P2.af2(1:Length/2+1);
                P1.af2(2:end-1) = 2*P1.af2(2:end-1);

                sensordata.df2{i}{j,k}(m,:) = fft(sensordata.d2{i}{j,k}(m,:), n_fft);
                P2.df2 = abs(sensordata.df2{i}{j,k}(m,:)/Length);
                P1.df2 = P2.df2(1:Length/2+1);
                P1.df2(2:end-1) = 2*P1.df2(2:end-1);

                id_row = i + num.shape*(num.etr*(j-1)+(k-1));
                [stats_at1,peak_factor_at1,~,~] = Signal_features(sensordata.a1{i}{j,k}(m,:));
                corr_at1 = corr(sensordata.a1{i}{j,k}(m,:)', dataset_shifted{i,(j-1)*num.etr+k}(1,:)');
                [stats_dt1,peak_factor_dt1,~,~] = Signal_features(sensordata.d1{i}{j,k}(m,:));
                corr_dt1 = corr(sensordata.d1{i}{j,k}(m,:)', dataset_shifted{i,(j-1)*num.etr+k}(1,:)');

                [stats_af1,peak_factor_af1,sc_af1,ss_af1] = Signal_features(P1.af1);
                [stats_df1,peak_factor_df1,sc_df1,ss_df1] = Signal_features(P1.df1);

                featurepool(id_row, num_fea_cloumn*2*(m-1)+(1:num_fea_cloumn)) = ...
                    [stats_at1,peak_factor_at1,corr_at1, ...
                     stats_dt1,peak_factor_dt1,corr_dt1, ...
                     stats_af1,peak_factor_af1,sc_af1,ss_af1, ...
                     stats_df1,peak_factor_df1,sc_df1,ss_df1];

                [stats_at2,peak_factor_at2,~,~] = Signal_features(sensordata.a2{i}{j,k}(m,:));
                corr_at2 = corr(sensordata.a2{i}{j,k}(m,:)', dataset_shifted{i,(j-1)*num.etr+k}(2,:)');
                [stats_dt2,peak_factor_dt2,~,~] = Signal_features(sensordata.d2{i}{j,k}(m,:));
                corr_dt2 = corr(sensordata.d2{i}{j,k}(m,:)', dataset_shifted{i,(j-1)*num.etr+k}(2,:)');

                [stats_af2,peak_factor_af2,sc_af2,ss_af2] = Signal_features(P1.af2);
                [stats_df2,peak_factor_df2,sc_df2,ss_df2] = Signal_features(P1.df2);

                featurepool(id_row, num_fea_cloumn*(2*m-1)+(1:num_fea_cloumn)) = ...
                    [stats_at2,peak_factor_at2,corr_at2, ...
                     stats_dt2,peak_factor_dt2,corr_dt2, ...
                     stats_af2,peak_factor_af2,sc_af2,ss_af2, ...
                     stats_df2,peak_factor_df2,sc_df2,ss_df2];
            end

            featurepool(id_row, num_fea_cloumn*2*m+(1:num_fea_row)) = ...
                featurepool(id_row, [num_fea_cloumn*(0:2:2*(N_wavlayer-1))+39, num_fea_cloumn*(0:2:2*(N_wavlayer-1))+54]) ...
               /sum(featurepool(id_row, [num_fea_cloumn*(0:2:2*(N_wavlayer-1))+39, num_fea_cloumn*(0:2:2*(N_wavlayer-1))+54]));

            featurepool(id_row, num_fea_cloumn*2*m+(num_fea_row+1:2*num_fea_row)) = ...
                featurepool(id_row, [num_fea_cloumn*(1:2:(2*N_wavlayer-1))+39, num_fea_cloumn*(1:2:(2*N_wavlayer-1))+54]) ...
               /sum(featurepool(id_row, [num_fea_cloumn*(1:2:(2*N_wavlayer-1))+39, num_fea_cloumn*(1:2:(2*N_wavlayer-1))+54]));

            [stats_sg1,peak_factor_sg1,~,~] = Signal_features(dataset_shifted{i,(j-1)*num.etr+k}(3,:));
            featurepool(id_row, num_fea_cloumn*2*m+num_fea_row*2+(1:13)) = [stats_sg1,peak_factor_sg1];

            [stats_sg2,peak_factor_sg2,~,~] = Signal_features(dataset_shifted{i,(j-1)*num.etr+k}(4,:));
            featurepool(id_row, num_fea_cloumn*2*m+num_fea_row*2+(14:26)) = [stats_sg2,peak_factor_sg2];
        end
    end
end
out_trial    = ones(num.group*num.trial*num.etr,1) * (1:num.shape);
output_class = reshape(out_trial', [], 1);

meta.readme      = readme;
meta.wavf        = wavf;
meta.N_wavlayer  = N_wavlayer;
meta.Num_point   = Num_point;
meta.n_fft       = n_fft;
meta.harmonics   = harmonics;
meta.Q           = Q;
meta.fs          = fs;
meta.num         = num;
assert(size(featurepool,2)==num.feature, 'Unexpected feature-pool dimension.');

if any(~isfinite(featurepool(:)))
    warning('featurepool 中存在 NaN/Inf，请检查输入数据是否有空 cell 或异常。');
end

end 


function [stats, peak_factor, spectral_centroid, spectral_spread] = Signal_features(x)
    stats(1)=mean(x);
    stats(2)=median(x);
    stats(3)=mode(x);
    stats(4)=range(x);
    stats(5)=sum(abs(x-mean(x)))/length(x);
    stats(6)=std(x);
    stats(7)=std(x)/mean(x);
    stats(8)=quantile(x,0.75)- quantile(x,0.25);
    stats(9)=skewness(x);
    stats(10)=kurtosis(x);
    stats(11)=sum(x.^2);

    signal=x;
    N = 256;
    quantized_signal = round((signal - min(signal)) / (max(signal) - min(signal)) * (N - 1)) + 1;
    unique_values = unique(quantized_signal);
    probabilities = histcounts(quantized_signal, unique_values) / numel(quantized_signal);
    info_entropy = -sum(probabilities .* log2(probabilities + eps));
    stats(12)=info_entropy;

    peak_value = max(signal);
    rms_value  = sqrt(mean(signal.^2));
    peak_factor = peak_value / rms_value;

    fs=1000;
    n_fft = length(x);
    freq_vector = (0:n_fft-1) * (fs / n_fft);
    spectral_centroid = sum(freq_vector .* (x / sum(x)));
    deviations = (freq_vector - spectral_centroid) .^ 2;
    spectral_spread = sqrt(sum(deviations .* (x / sum(x))));
end
