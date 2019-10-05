classdef GPFA
    %GPFA class wrapping the parameters and state of a GPFA model.
    
    properties
        %% --- Matrices ---
        Y % [T x N] data points
        C % [N x L] latents loadings
        R % [N x 1] private variance of each neuron (full covariance is diag(R))
        S % [T x M] stimulus values for linear-tuning model (optional - may be empty)
        D % [N x M] stimulus loadings for linear-tuning model (optional - requires S)
        Sf% [T x Mf] array or cell array of [T x Mf_k] stimuli for GP regression, where the kth term has dimension Mf_k (optional - may be empty)
        b % [N x 1] bias giving baseline firing rate of each neuron
        times % [1 x T] actual time points (use either 'times' or 'dt', not both)
        dt % [scalar] interval between trials if equi-spaced (use either 'times' or 'dt', not both)
        %% --- Sizes ---
        L % Number of latents
        T % Number of time points
        N % Number of neurons
        M % Number of stimulus conditions
        %% --- Kernel Parameters ---
        taus % [1 x L] timescale of each latent
        sigs % [1 x L] slow variability of each latent
        rhos % [1 x L] additional instantaneous variability of each latent
        taus_alpha % [1 x L] alpha (shape) parameter of gamma prior on taus
        taus_beta  % [1 x L] beta (rate) parameter of gamma prior on taus
        rho_scale % mean of exponential prior on 'rho' values
        %% --- Stimulus-GP Kernel Parameters ---
        stim_dist_fun % Function or cell array of functions taking in ([1 x Mf_k], [1 x Mf_k]) pairs of stimuli and returning a non-negative 'distance' between them
        signs % [1 x N] or cell array of [1 x N] variance of stimulus dependence per neuron (essentially quantifies net stimulus modulation)
        tauf  % scalar or [1 x K] array, length scale parameter for GP tuning
        forceZeroF % scalar logical or [1 x K] logical array: whether to force f(0)=0 for each stimulus dimension
        %% --- EM settings ---
        kernel_update_freq % how often (number of iterations) to update the kernel parameters
        fixed % Cell array of fixed parameter names. May specify sub indices, for instance 'C(:,1:5)' would specify that the first 5 latent loadings are fixed.
        lr    % learning rate for gradient-based updates
        lr_decay  % half-life of learning rate for simulated annealing
        useGPU % flag indicating whether to use GPU accelration on matrix operations
    end
    
    properties%(Access = protected)
        %% --- Precomputed matrices ---
        K     % [1 x L] cell array of prior covariances, each [T x T]
        Gamma % [L x L] cell array of data-dependent part of latent covariance matrices, each [T x T]. (Off-diagonal captures explaining away between latents)
        Cov   % [1 x L] cell array of posterior covariance matrices, inv(inv(K) + Gamma), computed stably for the case when inv(K) is poorly conditioned
        %% --- Useful things for stimulus GP tuning ---
        uSf    % Unique values of Sf, per row (cell array, 1 per each stimulus k)
        Sf_ord % Ordinal values for GP stimuli Sf, usable as indices into Ns and Kf. Hence Sf=uSf(Sf_ord,:) (cell array, 1 per each stimulus k)
        Ns     % [N x Mf], Ns(n,i) contains the number of trials where stimulus i appeared for neuron n, corresponding to unique values of Sf_ord (cell array, 1 per each stimulus k)
        Kf     % Precomputed GP Kernel for stimulus tuning (cell array, 1 per each stimulus k)
        ss2    % Squared pairwise distance function between stimuli (cell array, 1 per each stimulus k)
        nGP    % Number of GP tuning functions per neuron (i.e. number of independent stimulus dimensions)
    end
    
    properties (Access = private)
        initialized % whether updateAll() has been called to populate the large [T x T] matrices
    end
    
    methods
        %% Constructor and initialization
        function gpfaObj = GPFA(varargin)
            %GPFA create an instance of a GPFA class. Only required arguments are data 'Y' and
            %number of latents 'L':
            %
            %gpfa = GPFA(Y, L)
            %gpfa = GPFA('Y', Y, 'L', L)
            %
            %Any other fields of the GPFA class may optionally be set using analogous key-value
            %pairs.
            
            GPFA.ensureUtilPath();
            
            if nargin < 2
                error('Not enough inputs - must specify at least GPFA(Y, L)');
            end
            
            argStart = 1;
            requiredArgs = {'Y', 'L'};
            if ~ischar(varargin{1})
                gpfaObj.Y = varargin{1};
                argStart = 2;
                requiredArgs = {'L'};
                
                if ~ischar(varargin{2})
                    gpfaObj.L = varargin{2};
                    argStart = 3;
                    requiredArgs = {};
                end
            end
            
            %% Ensure that Y and L are given somewhere
            missingArgs = setdiff(requiredArgs, varargin(cellfun(@(v) ischar(v), varargin)));
            if ~isempty(missingArgs)
                error('Missing reqiured input(s): %s', strjoin(missingArgs, ', '));
            end
            
            %% Until one of the large matrices is accessed, we leave them all unitialized
            gpfaObj.initialized = false;
            
            %% Get all other fields from varargin and initialize everything
            gpfaObj = gpfaObj.setFields(varargin{argStart:end});
        end
        
        function gpfaObj = setFields(gpfaObj, varargin)
            %% Ensure no protected fields are being written
            % TODO - is there an introspective programmatic way to get these?
            protectedFields = {'K', 'Gamma', 'Cov', 'uSf', 'Sf_ord', 'Ns', 'Kf', 'ss2', 'nGP'};
            
            %% Copy fields from varargin
            allProps = properties(gpfaObj);
            for i=1:2:length(varargin)
                fieldname = varargin{i};
                % Note: isfield() does not work on objects, but searching for property names does.
                if ismember(fieldname, protectedFields)
                    warning('Refusing to set protected field ''%s''', fieldname);
                elseif any(strcmp(allProps, fieldname))
                    gpfaObj.(fieldname) = varargin{i+1};
                else
                    error('Unrecognized field: ''%s''', fieldname);
                end
            end
            
            %% Ensure that Y and L were given
            if isempty(gpfaObj.Y)
                error('Not enough inputs - Y is a required argument');
            end
            
            if isempty(gpfaObj.L)
                error('Not enough inputs - L is a required argument');
            end
            
            % Up front, compute mask of where 'missing' data are
            missing_data = isnan(gpfaObj.Y);
            
            %% Store and check matrix size consistency
            if isempty(gpfaObj.T), gpfaObj.T = size(gpfaObj.Y, 1); end
            if isempty(gpfaObj.N), gpfaObj.N = size(gpfaObj.Y, 2); end
            
            assert(isempty(gpfaObj.C) || all(size(gpfaObj.C) == [gpfaObj.N gpfaObj.L]), '''C'' must be size [N x L]');
            assert(isempty(gpfaObj.R) || all(size(gpfaObj.R) == [gpfaObj.N 1]), '''R'' must be size [N x 1]');
            assert(isempty(gpfaObj.b) || all(size(gpfaObj.b) == [gpfaObj.N 1]), '''b'' must be size [N x 1]');
            assert(isempty(gpfaObj.times) || length(gpfaObj.times) == gpfaObj.T, '''times'' must be size [1 x T]');
            assert(isempty(gpfaObj.dt) || isscalar(gpfaObj.dt), '''dt'' must be a scalar');
            
            if ~isempty(gpfaObj.S)
                if isempty(gpfaObj.M), gpfaObj.M = size(gpfaObj.S, 2); end
                assert(size(gpfaObj.S, 1) == gpfaObj.T, '''S'' must be size [T x M]');
                assert(isempty(gpfaObj.D) || all(size(gpfaObj.D) == [gpfaObj.N gpfaObj.M]), '''D'' must be size [N x M]');
            end
            
            if ~isempty(gpfaObj.Sf)
                if iscell(gpfaObj.Sf)
                    for k=1:length(gpfaObj.Sf)
                        assert(size(gpfaObj.Sf{k}, 1) == gpfaObj.T, '''Sf'' must be size [T x Mf]');
                    end
                else
                    assert(size(gpfaObj.Sf, 1) == gpfaObj.T, '''Sf'' must be size [T x Mf]');
                    gpfaObj.Sf = {gpfaObj.Sf};
                end
            
                if isempty(gpfaObj.forceZeroF)
                    gpfaObj.forceZeroF = false(1, length(gpfaObj.Sf));
                end
            end
            
            if isempty(gpfaObj.times) && isempty(gpfaObj.dt)
                gpfaObj.dt = 1;
                
                if ~isempty(gpfaObj.taus)
                    warning('''taus'' provided but no times were set!');
                end
            end
            
            assert(isempty(gpfaObj.taus) || length(gpfaObj.taus) == gpfaObj.L, '''taus'' must be length L');
            assert(isempty(gpfaObj.sigs) || length(gpfaObj.sigs) == gpfaObj.L, '''sigs'' must be length L');
            assert(isempty(gpfaObj.rhos) || length(gpfaObj.rhos) == gpfaObj.L, '''rhos'' must be length L');
            
            %% Default kernel parameters
            
            if isempty(gpfaObj.dt)
                effectiveDt = nanmean(diff(gpfaObj.times));
            else
                effectiveDt = gpfaObj.dt;
                gpfaObj.times = (1:gpfaObj.T) * gpfaObj.dt;
            end
            
            if isempty(gpfaObj.taus)
                gpfaObj.taus = 10 * effectiveDt * ones(1, gpfaObj.L);
            end
            
            % By default, taus_alpha and taus_beta form an exponential prior with a large mean
            % (as near as we can get to a flat prior)
            if isempty(gpfaObj.taus_alpha)
                gpfaObj.taus_alpha = ones(1, gpfaObj.L);
            end
            
            if isempty(gpfaObj.taus_beta)
                gpfaObj.taus_beta = ones(1, gpfaObj.L) / 100;
            end
            
            if isempty(gpfaObj.sigs)
                gpfaObj.sigs = ones(1, gpfaObj.L);
            end
            
            if isempty(gpfaObj.rho_scale)
                gpfaObj.rho_scale = 1e-3 * ones(1, gpfaObj.L);
            elseif length(gpfaObj.rho_scale) == 1
                gpfaObj.rho_scale = gpfaObj.rho_scale * ones(1, gpfaObj.L);
            end
            
            if isempty(gpfaObj.rhos)
                % Note: small nonzero rho helps numerical stability
                gpfaObj.rhos = gpfaObj.rho_scale;
            end
            
            %% Precompute things for stimulus tuning
            
            if ~isempty(gpfaObj.Sf)
                if ~iscell(gpfaObj.Sf), gpfaObj.Sf = {gpfaObj.Sf}; end
                gpfaObj.nGP = length(gpfaObj.Sf);
                
                if isempty(gpfaObj.tauf), gpfaObj.tauf = ones(1, gpfaObj.nGP); end
                if isempty(gpfaObj.signs), gpfaObj.signs = ones(gpfaObj.nGP, gpfaObj.N); end
                if ~iscell(gpfaObj.stim_dist_fun), gpfaObj.stim_dist_fun = {gpfaObj.stim_dist_fun}; end

                if isscalar(gpfaObj.forceZeroF), gpfaObj.forceZeroF = gpfaObj.forceZeroF*ones(1, gpfaObj.nGP); end
                
                % Preprocess each of the K GP stimulus sets
                for k=gpfaObj.nGP:-1:1
                    gpfaObj.uSf{k} = unique(gpfaObj.Sf{k}, 'rows');
                    dim_f = size(gpfaObj.uSf{k}, 1);
                    
                    if isempty(gpfaObj.stim_dist_fun{k})
                        warning('Sf but no stim_dist_fun given; naively assuming euclidean distances!');
                        gpfaObj.stim_dist_fun{k} = @(s1, s2) sqrt(sum((s1-s2).^2));
                    end
                    
                    ss = zeros(dim_f, dim_f);
                    gpfaObj.Ns{k} = zeros(gpfaObj.N, dim_f);
                    for iStim=1:dim_f
                        matches = all(gpfaObj.Sf{k} == gpfaObj.uSf{k}(iStim, :), 2);
                        gpfaObj.Ns{k}(:, iStim) = sum(matches & ~missing_data, 1)';
                        
                        for jStim=1:dim_f
                            ss(iStim, jStim) = gpfaObj.stim_dist_fun{k}(gpfaObj.uSf{k}(iStim, :), gpfaObj.uSf{k}(jStim, :));
                        end
                    end
                    
                    gpfaObj.ss2{k} = ss.^2;
                    
                    assert(all(all(gpfaObj.ss2{k} == gpfaObj.ss2{k}')), 'stim_dist_fun must be symmetric!');
                    
                    [~, gpfaObj.Sf_ord{k}] = ismember(gpfaObj.Sf{k}, gpfaObj.uSf{k}, 'rows');
                end
            end

            %% Defaults for EM settings
            
            if isempty(gpfaObj.fixed), gpfaObj.fixed = {}; end
            if isempty(gpfaObj.lr), gpfaObj.lr = 0.001; end
            if isempty(gpfaObj.lr_decay), gpfaObj.lr_decay = 100; end
            if isempty(gpfaObj.kernel_update_freq), gpfaObj.kernel_update_freq = 1; end
            if isempty(gpfaObj.useGPU), gpfaObj.useGPU = false; end
            
            gpfaObj.fixed = sort(gpfaObj.fixed);

            % Verify that GPU is available if selected
            if gpfaObj.useGPU
                try
                    G = gpuArray(eye(3));
                catch e
                    warning('useGPU was set to true, but GPU is not available. Message:\n%s', getReport(e));
                    gpfaObj.useGPU = false;
                end
            end
            
            %% Initialize loadings if they were not provided
            
            gpfaInit = gpfaObj.initialize();
            
            if isempty(gpfaObj.b), gpfaObj.b = gpfaInit.b; end
            if isempty(gpfaObj.C), gpfaObj.C = gpfaInit.C; end
            if isempty(gpfaObj.D), gpfaObj.D = gpfaInit.D; end
            if isempty(gpfaObj.R), gpfaObj.R = gpfaInit.R; end
            
            % Initialize all 'precomputed' matrices if they're already in memory anyway
            if gpfaObj.initialized
                gpfaObj = gpfaObj.updateAll();
            end
        end
        
        function gpfaObj = populate(gpfaObj)
            if ~gpfaObj.initialized
                gpfaObj = gpfaObj.updateAll();
            end
        end
        
        %% Inference
        [mu_x, sigma_x] = inferX(gpfaObj, queryTimes)
        [mu_x, sigma_x, mu_f, sigma_f] = inferMeanFieldXF(gpfaObj, queryTimes, queryStims, maxIters, convTol)
        
        function [fve, fve_per_unit] = fve(gpfaObj)
            %GPFA.FVE fraction variance explained
            per_unit_var = nanvar(gpfaObj.Y, [], 1);
            resid = gpfaObj.Y - gpfaObj.predictY();
            per_unit_resid_var = nanvar(resid, [], 1);
            
            per_unit_var_explained = per_unit_var - per_unit_resid_var;
            fve_per_unit = per_unit_var_explained ./ per_unit_var;
            
            fve = mean(fve_per_unit);
        end
        
        %% Learning
        [gpfaObj, Q, H] = emStep(gpfaObj, itr, infer_tol)
        [bestFit, Qs, Hs, FVEs, converged] = fitEM(gpfaObj, maxIters, convergenceTol, startIter)
        
        %% Simulation / Generate Data
        [Yhat, x, f] = simulate(gpfaObj, x, f)
        [mu_Y, mu_Ysq] = predictY(gpfaObj, mu_x, mu_f)
        [Y] = sampleY(gpfaObj, nSamples, mu_x, sigma_x, mu_f, sigma_f)
        
        %% Save to disk while reasonably managing file sizes
        function gpfaObj = saveobj(gpfaObj)
            % Remove large fields that can be reconstructed after re-loading, and indicate this by
            % setting the 'initialized' flag to false. Also, flag 'useGPU' as false since data may
            % be saved from a CUDA-enabled machine and loaded from a CPU-only machine
            gpfaObj = gpfaObj.clearMatrices();
            gpfaObj.useGPU = false;
        end
        
        function gpfaObj = clearMatrices(gpfaObj)
            gpfaObj.initialized = false;
            gpfaObj.K = {};
            gpfaObj.Gamma = {};
            gpfaObj.Cov = {};
        end
    end
    
    methods (Access = protected)
        %% Helper to initialize parameters based on data
        function gpfaObj = initialize(gpfaObj)
            % An unfortunate misnomer: the initialize() function does not set initialized=True. This
            % function is responsible for initializing everything *except* the large [T x T] arrays.
            % Initializing those arrays (and setting initialized=True) is done inside
            % GPFA.updateAll()
            
            % Initialize mean b using mean of data
            if ~any(strcmp('b', gpfaObj.fixed))
                gpfaObj.b = nanmean(gpfaObj.Y, 1)';
            end
            
            % Initialize stimulus loadings D using linear regression
            if ~any(strcmp('D', gpfaObj.fixed)) && ~isempty(gpfaObj.S)
                % The following is the same as first replacing each missing value with the mean
                % (since b is the mean) then regressing using (Y-b)/S
                Yeffective = gpfaObj.Y' - gpfaObj.b;
                Yeffective(isnan(Yeffective)) = 0;
                % mrdivide with [N x T] / [M x T] results in a [N x M] matrix D
                gpfaObj.D = Yeffective / gpfaObj.S';
            end
            
            residuals = gpfaObj.Y - gpfaObj.b';
            if ~isempty(gpfaObj.S)
                residuals = residuals - gpfaObj.S * gpfaObj.D';
            end
            
            % Initialize latent loadings C by factor analysis
            if gpfaObj.L == 0
                gpfaObj.C = [];
            elseif ~any(strcmp('C', gpfaObj.fixed))
                gpfaObj = gpfaObj.updateK();
                if any(isnan(gpfaObj.Y(:)))
                    % For initialization of loadings with missing data, impute missing values and
                    % hope for the best.. Errors in this process will be corrected by fitEM() later.
                    loading_residuals = impute(residuals);
                else
                    loading_residuals = residuals;
                end
                
                [~, isrt] = sort(gpfaObj.taus, 'descend');
                
                initC = zeros(gpfaObj.N, gpfaObj.L);
                for iTau=1:gpfaObj.L
                    idx = isrt(iTau);
                    tau = gpfaObj.taus(idx);
                    kernel = exp(-0.5*(gpfaObj.times - gpfaObj.times').^2 / tau^2);
                    kernel = kernel ./ sum(kernel, 2);
                    % This dot product is [T x T] x [T x N]; it averages together data with a
                    % 'window' that depends on the kernel size and time points.
                    dataSmooth = kernel * loading_residuals;
                    % Use top principal components of 'smoothed' data to initialize loadings
                    smoothCov = nancov(dataSmooth, 'pairwise');
                    [initC(:,idx), ~] = eigs(smoothCov, 1);
                    % To continue to next iteration, first subtract off this component (note that
                    % initC is currently a unit vector from eigs function)
                    est_x = kernel * loading_residuals * initC(:,idx);
                    loading_residuals = loading_residuals - est_x .* initC(:,idx)';
                    % Scale the loadings so the inferred x will have variancenear the expected
                    % sample variance at this timescale.
                    covK = gpfaObj.K{idx};
                    % Analytic sample variance, i.e. E[(x_t - <x>)^2/T]
                    sample_variance_x = (sum(diag(covK))+(1/gpfaObj.T^2-1/gpfaObj.T)*sum(covK(:)))/gpfaObj.T;
                    scale = sqrt(var(est_x) / sample_variance_x);
                    initC(:,idx) = initC(:,idx) * scale;
                end
                gpfaObj.C = initC;
            end
            
            % Initialize private variance R using residuals from the stimulus prediction only
            if ~any(strcmp('R', gpfaObj.fixed))
                gpfaObj.R = max(nanvar(residuals, [], 1)', 1e-3);
            end
        end
        
        %% Derivative and Q function value w.r.t. GP scales
        function Q = timescaleQ(gpfaObj, mu_x, cov_x)
            if ~gpfaObj.initialized, gpfaObj = gpfaObj.updateAll(); end
            Q = 0;
            for l=1:gpfaObj.L
                tau = gpfaObj.taus(l);
                rho = gpfaObj.rhos(l);
                alph = gpfaObj.taus_alpha(l);
                beta = gpfaObj.taus_beta(l);
                % Get prior covariance matrix
                Kl = gpfaObj.K{l};
                Kli = inv(Kl);
                % Compute prior values
                log_prior_rho = -rho / gpfaObj.rho_scale(l);
                log_prior_tau = alph*log(beta) + (alph-1)*log(tau) - beta*tau - gammaln(alph);
                % Compute Q for latent l
                Q = Q - 1/2*(mu_x(:,l)'*Kli*mu_x(:,l) + tracedot(cov_x{l}, Kli) + logdet(2*pi*Kl)) ...
                    + log_prior_rho + log_prior_tau; %#ok<MINV>
            end
            Q = double(gather(Q));
        end
        
        function [dQ_dlogtau2, dQ_dlogrho2] = timescaleDeriv(gpfaObj, mu_x, cov_x)
            if ~gpfaObj.initialized, gpfaObj = gpfaObj.updateAll(); end
            dQ_dlogtau2 = zeros(size(gpfaObj.taus));
            dQ_dlogrho2 = zeros(size(gpfaObj.taus));
            dt2 = (gpfaObj.times - gpfaObj.times').^2;
            for l=1:gpfaObj.L
                tau = gpfaObj.taus(l);
                rho = gpfaObj.rhos(l);
                alph = gpfaObj.taus_alpha(l);
                beta = gpfaObj.taus_beta(l);
                % Get K and E[xx'] matrices
                Kl = gpfaObj.K{l};
                Kli = inv(Kl);
                e_xx_l = mu_x(:,l) .* mu_x(:,l)' + cov_x{l};
                % Compute derivatives
                dQ_dKl = Kli * e_xx_l * Kli - Kli; %#ok<MINV>
                dt2_div_tau2 = dt2./(2*tau^2);
                dKl_dlogtaul2 = 0.5 * gpfaObj.sigs(l)^2 * exp(-dt2_div_tau2) .* dt2_div_tau2;
                % Matrix chain rule plus prior gradient. Note that since MAP is not invariant to
                % reparameterization, the gradients here deliberately do not take into account the
                % |dtau/dlogtau2| change-of-variables term. The upshot is that we are taking
                % gradients with respect to log(tau^2) towards finding the MAP point with respect to
                % plain old tau.
                dPrior_dlogtau2 = (alph - 1 - beta*tau)/2;
                dQ_dlogtau2(l) = gather(sum(sum(dQ_dKl .* dKl_dlogtaul2))) + dPrior_dlogtau2;
                % Rho is easier since it doesn't depend on time differences; include a derivative on
                % the prior
                dQ_dlogrho2(l) = gather(sum(diag(dQ_dKl)))*rho^2 - rho/(2*gpfaObj.rho_scale(l));
            end
        end
        
        function dQ_dlogtauf2 = stimScaleDeriv(gpfaObj, mu_f, sigma_f)
            if ~gpfaObj.initialized, gpfaObj = gpfaObj.updateAll(); end
            dQ_dlogtauf2 = zeros(size(gpfaObj.tauf));
            for k=gpfaObj.nGP:-1:1
                dK_dlogtau2f = gpfaObj.Kf{k} .* gpfaObj.ss2{k} / gpfaObj.tauf(k)^2;
                dimf = size(gpfaObj.Kf{k}, 1);
                Ki = gpfaObj.Kf{k} \ speye(dimf);
                for n=1:gpfaObj.N
                    sign = gpfaObj.signs(k,n);
                    e_ff_n = mu_f{k}(:,n)*mu_f{k}(:,n)' + sigma_f{k}{n};
                    dQfn_dK = sign^(2*dimf)*Ki - 1/sign^2 * Ki * e_ff_n * Ki;
                    dQ_dlogtauf2(k) = dQ_dlogtauf2(k) - 1/2*tracedot(dQfn_dK, dK_dlogtau2f);
                end
            end
        end
        
        %% Functions to update 'precomupted' terms when underlying parameters change
        function gpfaObj = updateK(gpfaObj)
            if gpfaObj.L == 0
                gpfaObj.K = {};
                return
            end
            
            % Create array of timepoints for each measurement, either from 'dt' or simply use
            % 'times'
            if ~isempty(gpfaObj.dt)
                ts = gpfaObj.dt * (1:gpfaObj.T);
            elseif ~isempty(gpfaObj.times)
                ts = gpfaObj.times;
            else
                error('Need either ''times'' or ''dt''');
            end

            % Transfer O(T) time points to GPU memory so that O(T^2) update of K itself happens on
            % the GPU
            if gpfaObj.useGPU
                ts = gpuArray(ts);
            end
            
            timeDiffs2 = (ts - ts').^2;
            
            for l=1:gpfaObj.L
                sig = gpfaObj.sigs(l);
                tau = gpfaObj.taus(l);
                rho = gpfaObj.rhos(l);
                gpfaObj.K{l} = sig^2 * exp(-timeDiffs2 / (2 * tau^2)) + rho^2 * speye(size(timeDiffs2));
            end
        end
        
        function gpfaObj = updateGamma(gpfaObj, Y)
            if gpfaObj.L == 0
                gpfaObj.Gamma = {};
                return
            end
            
            if ~exist('Y', 'var'), Y = gpfaObj.Y; end
            
            if ~any(isnan(Y(:)))
                % CRiC is C'*inv(R)*C but we have R as the elements of a diagonal...
                CRiC = gpfaObj.C' * (gpfaObj.C ./ gpfaObj.R);
                for l1=1:gpfaObj.L
                    for l2=1:gpfaObj.L
                        gpfaObj.Gamma{l1, l2} = CRiC(l1, l2) * speye(gpfaObj.T);
                    end
                end
            else
                valid = ~isnan(Y);
                % validRi is [N x T] and contains elements of R-inverse wherever there is valid data.
                validRi = valid' ./ gpfaObj.R;
                % partialCC is [L x L x N]. It is like C'*C without taking the sum over the N dimension yet.
                partialCC = reshape(gpfaObj.C', gpfaObj.L, 1, gpfaObj.N) .* reshape(gpfaObj.C', 1, gpfaObj.L, gpfaObj.N);
                % Final result is [L x L x T]. It will be reshaped into the full Gamma matrix later.
                gammaDense = reshape(sum(partialCC .* reshape(validRi, 1, 1, gpfaObj.N, gpfaObj.T), 3), [gpfaObj.L, gpfaObj.L, gpfaObj.T]);
                % Allocate space for [L x L] cell array of diagonal matrices
                gpfaObj.Gamma = cell(gpfaObj.L);
                for l1=1:gpfaObj.L
                    for l2=1:gpfaObj.L
                        gpfaObj.Gamma{l1, l2} = spdiag(squeeze(gammaDense(l1, l2, :)));
                    end
                end
            end
                
            % NOTE: as of version 2019a, Matlab's support for sparse gpuArrays is still limited. We
            % could either use a full gpuArray or a sparse normal array for Gamma. It seems to be a
            % better idea to do O(T) operations can happen on the CPU than to transfer O(T^2)
            % matrices back and forth from CPU to GPU memory. If Mathworks someday fixes this, the
            % following should be uncommented:

            % if gpfaObj.useGPU
            %     for l1=1:gpfaObj.L
            %         for l2=1:gpfaObj.L
            %             gpfaObj.Gamma{l1, l2} = gpuArray(gpfaObj.Gamma{l1, l2});
            %         end
            %     end
            % end
        end
        
        function gpfaObj = updateCov(gpfaObj)
            if gpfaObj.L == 0
                gpfaObj.Cov = {};
                return
            end
            
            % Compute posterior cov, inv(inv(K) + Gamma) using the following identity to avoid
            % actually taking inv(K):
            %   inv(A+B) = inv(A) - inv(A)*B*inv(I+inv(A)*B)*inv(A)
            % which, substituting inv(A) as K gives
            %   inv(inv(K) + G) = K - K * G * ((I + K * G) \ K)
            for l=1:gpfaObj.L
                k = gpfaObj.K{l};
                G = gpfaObj.Gamma{l, l};
                I = speye(size(G));
                gpfaObj.Cov{l} = k - k * G * ((I + k * G) \ k);
            end
        end
        
        function gpfaObj = updateKernelF(gpfaObj)
            % Note: Kf is only [S x S]. The prior covariance per neuron is Kf*signs(n)^2.
            % Adding a small diagonal component for stability
            for k=gpfaObj.nGP:-1:1
                gpfaObj.Kf{k} = exp(-gpfaObj.ss2{k} / gpfaObj.tauf(k)^2);
                gpfaObj.Kf{k} = gpfaObj.Kf{k} + 1e-6 * eye(size(gpfaObj.Kf{k}));
            end
        end
        
        function gpfaObj = updateAll(gpfaObj, Y)
            if exist('Y', 'var')
                gpfaObj = updateCov(updateK(updateGamma(gpfaObj, Y)));
            else
                gpfaObj = updateCov(updateK(updateGamma(gpfaObj)));
            end
            if ~isempty(gpfaObj.Sf)
                gpfaObj = updateKernelF(gpfaObj);
            end
            
            gpfaObj.initialized = true;
        end
        
        function [newValue, updateMask] = getNewValueHandleConstraints(gpfaObj, paramName, setValue)
            %GPFA.GETNEWVALUEHANDLEFIXED without actually setting the value, find what the new value
            %of a field would be if we tried to set it, taking into account all constraints
            %specified in 'gpfaObj.fixed'. Possible future optimization: cache results.
            oldValue = gpfaObj.(paramName);
            newValue = setValue;
            isConstrained = any(cellfun(@(fixedstring) startsWith(fixedstring, paramName), gpfaObj.fixed));
            
            updateMask = true(size(oldValue));

            if isConstrained
                for iConstraint=1:length(gpfaObj.fixed)
                    % If the constraint is an entire parameter like 'b', then 'parts' will be a cell
                    % array containing only {'b'}. If it has subscripts, like 'b(1)', then 'parts'
                    % will contain two entries, {'b', '1)'}
                    parts = strsplit(gpfaObj.fixed{iConstraint}, '(');
                    if strcmp(paramName, parts{1})
                        % Apply this constraint
                        if length(parts) == 1
                            % Simple constraint on the entire parameter. Can break after.
                            newValue = oldValue;
                            return
                        else
                            % Subscripted constraint. Apply matlab subscripting operations
                            
                            % Get the '...' out of 'b(...)'
                            insideParens = parts{2}(1:end-1);
                            
                            % Split by commas and strip whitespace. '(1, 2:3, :)' becomes {'1',
                            % '2:3', ':'}
                            subscripts = cellfun(@(substr) strip(substr), strsplit(insideParens, ','), ...
                                'UniformOutput', false);
                            
                            % Leave ':' intact but evaluate numeric indices
                            for i=1:length(subscripts)
                                if ~strcmp(subscripts{i}, ':')
                                    subscripts{i} = str2num(subscripts{i}); %#ok
                                end
                            end
                            
                            % Set update mask to false wherever indices of 'fixed' values are
                            % specified
                            substruct = struct('type', '()', 'subs', {subscripts});
                            updateMask = subsasgn(updateMask, substruct, false);
                        end
                    end
                end
                
                newValue(~updateMask) = oldValue(~updateMask);
            end
        end
    end
    
    methods (Static)
        function ensureUtilPath()
            if ~exist('spdiag', 'file')
                addpath('util');
            end
            if ~exist('blockinv', 'file')
                addpath(fullfile('util', 'block-matrix-inverse-tools'));
                addpath(fullfile('util', 'logdet'));
            end
        end
        
        function gpfaObj = loadobj(gpfaObj)
            % Sanity-check and backwards compatibility (see @saveobj)
            gpfaObj.initialized = false;
            gpfaObj.useGPU = false;
        end
    end
end
