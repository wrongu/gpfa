classdef GPFA
    %GPFA class wrapping the parameters and state of a GPFA model.
    
    properties
        %% --- Matrices ---
        Y % [T x N] data points
        C % [N x L] latents loadings
        R % [N x 1] private variance of each neuron (full covariance is diag(R))
        S % [T x M] stimulus values (optional - may be empty)
        Sf% [T x Mf] stimuli for GP stimulus regression (optional - may be empty)
        D % [N x M] stimulus loadings (optional - requires S)
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
        %% --- Stimulus-GP Kernel Parameters ---
        stim_dist_fun % Function taking in ([1 x Mf], [1 x Mf]) pairs of stimuli and returning a non-negative 'distance' between them
        signs % [1 x N] variance of stimulus dependence per neuron (essentially quantifies net stimulus modulation)
        tauf  % scalar length scale parameter for GP tuning
        %% --- EM settings ---
        fixed % cell array of fixed parameter names
        lr    % learning rate for gradient-based updates
        lr_decay  % half-life of learning rate for simulated annealing
        rho_scale % mean of exponential prior on 'rho' values
    end
    
    properties% (Access = protected)
        %% --- Metadata ---
        isKernelToeplitz   % whether K matrix has toeplitz structure (requires equal-spaced time points)
        % preTransform     % preprocessing applied to data (e.g. @sqrt to square-root transform spike counts)
        % postTransform    % the inverse of preTransform
        %% --- Precomputed matrices ---
        K     % [TL x TL] (sparse matrix) kernel-based covariance for (flattened) latents
        Gamma % [TL x TL] (sparse matrix) Kronecker of (C'*inv(R)*C) and eye(T), adjusted for missing data.
        Cov   % Posterior covariance matrix, inv(inv(K) + Gamma), computed stably for the case when inv(K) is poorly conditioned
        %% --- Useful things for stimulus GP tuning ---
        Sf_ord % Ordinal values for GP stimuli Sf, usable as indices into Ns and Kf
        Ns     % Ns(i) contains the number of trials where stimulus i appeared, corresponding to unique values of Sf_ord
        Kf     % Precomputed GP Kernel for stimulus tuning
        ss2    % Squared pairwise distance function between stimuli
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
            
            %% Get all other fields from varargin and initialize everything
            gpfaObj = gpfaObj.setFields(varargin{argStart:end});
        end
        
        function gpfaObj = setFields(gpfaObj, varargin)
            %% Ensure no protected fields are being written
            % TODO - is there an introspective programmatic way to get these?
            protectedFields = {'isKernelToeplitz', 'K', 'Gamma', 'Cov', 'Ns', 'ss2'};
            
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
            
            %% Store and check matrix size consistency
            if isempty(gpfaObj.T), gpfaObj.T = size(gpfaObj.Y, 1); end
            if isempty(gpfaObj.N), gpfaObj.N = size(gpfaObj.Y, 2); end
            
            assert(isempty(gpfaObj.C) || all(size(gpfaObj.C) == [gpfaObj.N gpfaObj.L]), '''C'' must be size [N x L]');
            assert(isempty(gpfaObj.R) || all(size(gpfaObj.R) == [gpfaObj.N 1]), '''R'' must be size [N x 1]');
            assert(isempty(gpfaObj.b) || all(size(gpfaObj.b) == [gpfaObj.N 1]), '''b'' must be size [N x 1]');
            assert(isempty(gpfaObj.times) || length(gpfaObj.times) == gpfaObj.T, '''times'' must be size [T x 1]');
            assert(isempty(gpfaObj.dt) || isscalar(gpfaObj.dt), '''dt'' must be a scalar');
            
            if ~isempty(gpfaObj.S)
                if isempty(gpfaObj.M), gpfaObj.M = size(gpfaObj.S, 2); end
                assert(size(gpfaObj.S, 1) == gpfaObj.T, '''S'' must be size [T x M]');
                assert(isempty(gpfaObj.D) || all(size(gpfaObj.D) == [gpfaObj.N gpfaObj.M]), '''D'' must be size [N x M]');
            end
            
            if ~isempty(gpfaObj.Sf)
                assert(size(gpfaObj.Sf, 1) == gpfaObj.T, '''Sf'' must be size [T x Mf]');
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
                effectiveDt = mean(diff(gpfaObj.times));
                gpfaObj.isKernelToeplitz = all(diff(gpfaObj.times) == effectiveDt);
            else
                effectiveDt = gpfaObj.dt;
                gpfaObj.isKernelToeplitz = true;
                gpfaObj.times = (1:gpfaObj.T)' * gpfaObj.dt;
            end
            
            if isempty(gpfaObj.taus)
                gpfaObj.taus = 10 * effectiveDt * ones(1, gpfaObj.L);
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
                        
            %% Check for and apply preprocessing transformations
            
            % if ~isempty(gpfaObj.preTransform)
            %     % Transform Y data
            %     gpfaObj.Y = gpfaObj.preTransform(gpfaObj.Y);
            % 
            %     % If not given, try to automatically infer what 'postTransform' should be.
            %     if isempty(gpfaObj.postTransform)
            %         if isequal(gpfaObj.preTransform, @sqrt)
            %             gpfaObj.postTransform = @(x) x.^2;
            %         elseif isequal(gpfaObj.preTransform, @log)
            %             gpfaObj.postTransform = @exp;
            %         else
            %             error('Not sure how to invert ''%s''. Supply your own ''postTransform''', func2str(gpfaObj.preTransform));
            %         end
            %     end
            % elseif ~isempty(gpfaObj.postTransform)
            %     error('''postTransform'' is given without any ''preTransform''');
            % end
            
            %% Precompute things for stimulus tuning
            
            if ~isempty(gpfaObj.Sf)
                uniqueStim = unique(gpfaObj.Sf, 'rows');
                dim_f = size(uniqueStim, 1);
                
                if isempty(gpfaObj.stim_dist_fun)
                    warning('Sf but no stim_dist_fun given; naively assuming euclidean distances!');
                    gpfaObj.stim_dist_fun = @(s1, s2) sqrt(sum((s1-s2).^2));
                end
                
                gpfaObj.Ns = zeros(dim_f, 1);
                for iStim=1:dim_f
                    matches = all(gpfaObj.Sf == uniqueStim(iStim, :), 2);
                    gpfaObj.Ns(iStim) = sum(matches);
                    gpfaObj.Sf_ord(matches) = iStim;
                    
                    for jStim=1:dim_f
                        gpfaObj.ss2(iStim, jStim) = gpfaObj.stim_dist_fun(uniqueStim(iStim, :), uniqueStim(jStim, :))^2;
                    end
                end
                
                assert(all(all(gpfaObj.ss2 == gpfaObj.ss2')), 'stim_dist_fun must be symmetric!');
                    
                % Correct for impossible distances that fail to satisfy the triangle inequality
                if any(isinf(gpfaObj.ss2(:)))
                    ss = sqrt(gpfaObj.ss2);
                    idxImpossible = find(triu(isinf(ss)));
                    for idx=idxImpossible'
                        [iStim, jStim] = ind2sub(size(ss), idx);
                        dist_i_x = ss(iStim, :);
                        dist_j_x = ss(jStim, :);
                        max_possible_dist = min(dist_i_x + dist_j_x);
                        ss(iStim, jStim) = max_possible_dist;
                        ss(jStim, iStim) = max_possible_dist;
                    end
                    gpfaObj.ss2 = ss.^2;
                end
                
                if isempty(gpfaObj.signs), gpfaObj.signs = ones(1, gpfaObj.N); end
                if isempty(gpfaObj.tauf), gpfaObj.tauf = 1; end
            end
            
            %% Initialize loadings if they were not provided
            
            if isempty(gpfaObj.fixed), gpfaObj.fixed = {}; end
            if isempty(gpfaObj.lr), gpfaObj.lr = 0.001; end
            if isempty(gpfaObj.lr_decay), gpfaObj.lr_decay = 100; end
            
            gpfaInit = gpfaObj.initialize();
            
            if isempty(gpfaObj.b), gpfaObj.b = gpfaInit.b; end
            if isempty(gpfaObj.C), gpfaObj.C = gpfaInit.C; end
            if isempty(gpfaObj.D), gpfaObj.D = gpfaInit.D; end
            if isempty(gpfaObj.R), gpfaObj.R = gpfaInit.R; end
            
            % Initialize all 'precomputed' matrices
            gpfaObj = gpfaObj.updateAll();
        end
        
        %% Inference
        [mu_x, sigma_x] = inferX(gpfaObj)
        [mu_x, sigma_x, mu_f, sigma_f] = inferMeanFieldXF(gpfaObj) % Joint inference of x with tuning curves
        
        %% Learning
        [gpfaObj, Q, H] = emStep(gpfaObj, itr)
        [bestFit, Qs, Hs] = fitEM(gpfaObj, maxIters, convergenceTol)
        
        %% Simulation / Generate Data
        [Yhat, x, f] = simulate(gpfaObj)
        [mu_Y] = predictY(gpfaObj, mu_x, mu_f)
        [Y] = sampleY(gpfaObj, nSamples, mu_x, sigma_x, mu_f, sigma_f)
    end
    
    methods (Access = protected)
        %% Helper to initialize parameters based on data
        function gpfaObj = initialize(gpfaObj)
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
            
            % Initialize latent loadings C using top L principal components of data after regressing
            % out the stimulus and smoothing by kernels of each unique 'tau'
            if ~any(strcmp('C', gpfaObj.fixed))                
                if ~any(isnan(gpfaObj.Y(:)))
                    uTaus = unique(gpfaObj.taus);
                    for i=1:length(uTaus)
                        kernel = exp(-0.5*(gpfaObj.times - gpfaObj.times').^2 / uTaus(i)^2);
                        kernel = kernel ./ sum(kernel, 2);
                        % This dot product is [T x T] x [T x N]; it averages together data with a
                        % 'window' that depends on the kernel size and time points.
                        dataSmooth = kernel * residuals;
                        % Use top principal components of 'smoothed' data to initialize loadings at this
                        % time-scale
                        smoothCov = nancov(dataSmooth, 'pairwise');
                        nTaus = sum(gpfaObj.taus == uTaus(i));
                        [gpfaObj.C(:, gpfaObj.taus == uTaus(i)), ~] = eigs(smoothCov, nTaus);
                    end
                else
                    % If there is missing data, the above smoothing method will likely fail.
                    % Initialize loadings randomly.
                    vars = nanvar(residuals, 1, 1);
                    scale = mean(sqrt(vars));
                    gpfaObj.C = scale * randn(gpfaObj.N, gpfaObj.L);
                end
            end
            
            % Initialize private variance R using residuals from the stimulus prediction only,
            % scaled up by 10 because over-estimating variance early helps keep EM stable.
            if ~any(strcmp('R', gpfaObj.fixed))
                gpfaObj.R = 10 * nanvar(residuals, [], 1)';
            end
        end

        %% Derivative and Q function value w.r.t. GP scales
        function [Q, dQ_dlogtau2, dQ_dlogrho2] = timescaleDeriv(gpfaObj, mu_x, cov_x)
            Q = 0;
            dQ_dlogtau2 = zeros(size(gpfaObj.taus));
            dQ_dlogrho2 = zeros(size(gpfaObj.taus));
            dt2 = (gpfaObj.times - gpfaObj.times').^2;
            for l=1:gpfaObj.L
                subs = (1:gpfaObj.T) + (l-1)*gpfaObj.T;
                Kl = gpfaObj.K(subs, subs);
                Kli = inv(Kl);
                e_xx_l = mu_x(:,l) .* mu_x(:,l)' + cov_x(subs, subs);
                log_prior_rho = -gpfaObj.rhos(l) / gpfaObj.rho_scale(l);
                Q = Q - 1/2*(mu_x(:,l)'*Kli*mu_x(:,l) + tracedot(cov_x(subs, subs), Kli) + logdet(2*pi*Kl)) + log_prior_rho; %#ok<MINV>
                dQ_dKl = Kli * e_xx_l * Kli - Kli; %#ok<MINV>
                dt2_div_tau2 = dt2./(2*gpfaObj.taus(l)^2);
                dKl_dlogtaul2 = 0.5 * gpfaObj.sigs(l)^2 * exp(-dt2_div_tau2) .* dt2_div_tau2;
                % Matrix chain rule
                dQ_dlogtau2(l) = dQ_dKl(:)' * dKl_dlogtaul2(:);
                % Rho is easier since it doesn't depend on time differences; include a derivative on
                % the prior
                dQ_dlogrho2(l) = sum(diag(dQ_dKl))*gpfaObj.rhos(l)^2 - gpfaObj.rhos(l)/(2*gpfaObj.rho_scale(l));
            end
        end
        
        function dQ_dlogtauf2 = stimScaleDeriv(gpfaObj, mu_f, sigma_f)
            dQ_dlogtauf2 = 0;
            dK_dlogtau2f = gpfaObj.Kf .* gpfaObj.ss2 / gpfaObj.tauf^2;
            % We can get NaN values here where ss2 is inf and thus Kf is 0.. Clearly the correct
            % semantics would be for the derivative of these entries to be zero.
            dK_dlogtau2f(isnan(dK_dlogtau2f)) = 0;
            dimf = size(gpfaObj.Kf, 1);
            Ki = gpfaObj.Kf \ speye(dimf);
            for n=1:gpfaObj.N
                sign = gpfaObj.signs(n);
                e_ff_n = mu_f(:,n)*mu_f(:,n)' + sigma_f{n};
                dQfn_dK = sign^(2*dimf)*Ki - 1/sign^2 * Ki * e_ff_n * Ki;
                dQ_dlogtauf2 = dQ_dlogtauf2 - 1/2*tracedot(dQfn_dK, dK_dlogtau2f);
            end
        end
        
        %% Functions to update 'precomupted' terms when underlying parameters change
        function gpfaObj = updateK(gpfaObj)
            Kcell = cell(1, gpfaObj.L);
            
            % Create array of timepoints for each measurement, either from 'dt' or simply use
            % 'times'
            if ~isempty(gpfaObj.dt)
                ts = gpfaObj.dt * (1:gpfaObj.T)';
            elseif ~isempty(gpfaObj.times)
                ts = gpfaObj.times(:);
            else
                error('Need either ''times'' or ''dt''');
            end
            
            timeDiffs2 = (ts - ts').^2;
            
            for l=1:gpfaObj.L
                sig = gpfaObj.sigs(l);
                tau = gpfaObj.taus(l);
                rho = gpfaObj.rhos(l);
                Kcell{l} = sig^2 * exp(-timeDiffs2 / (2 * tau^2)) + rho^2 * speye(gpfaObj.T);
            end
            
            gpfaObj.K = spblkdiag(Kcell{:});
        end
        
        function gpfaObj = updateGamma(gpfaObj, Y)
            if ~exist('Y', 'var'), Y = gpfaObj.Y; end
            
            if ~any(isnan(Y(:)))
                % CRiC is C'*inv(R)*C but we have R as the elements of a diagonal...
                CRiC = gpfaObj.C' * (gpfaObj.C ./ gpfaObj.R);
                gpfaObj.T = size(Y, 1);
                gpfaObj.Gamma = kron(CRiC, speye(gpfaObj.T));
            else
                valid = ~isnan(Y);
                % validRi is [N x T] and contains elements of R-inverse wherever there is valid data.
                validRi = valid' ./ gpfaObj.R;
                % partialCC is [L x L x N]. It is like C'*C without taking the sum over the N dimension yet.
                partialCC = reshape(gpfaObj.C', gpfaObj.L, 1, gpfaObj.N) .* reshape(gpfaObj.C', 1, gpfaObj.L, gpfaObj.N);
                % Final result is [L x L x T]. It will be reshaped into the full Gamma matrix later.
                gammaDense = reshape(sum(partialCC .* reshape(validRi, 1, 1, gpfaObj.N, gpfaObj.T), 3), [gpfaObj.L, gpfaObj.L, gpfaObj.T]);
                % Allocate space for [L x L] cell array of diagonal matrices
                gammaBlocks = cell(gpfaObj.L);
                for l1=1:gpfaObj.L
                    for l2=1:gpfaObj.L
                        gammaBlocks{l1, l2} = spdiag(squeeze(gammaDense(l1, l2, :)));
                    end
                end
                gpfaObj.Gamma = cell2mat(gammaBlocks);
            end
            
            % Sanity check
            assert(issparse(gpfaObj.Gamma));
        end
        
        function gpfaObj = updateCov(gpfaObj)
            % Compute posterior cov, inv(inv(K) + Gamma) using the following identity to avoid
            % actually taking inv(K):
            %   inv(A+B) = inv(A) - inv(A)*B*inv(I+inv(A)*B)*inv(A)
            % which, substituting inv(A) as K gives
            %   inv(inv(K) + G) = K - K * G * ((I + K * G) \ K)
            k = gpfaObj.K;
            G = gpfaObj.Gamma;
            I = speye(size(G));
            blocks = gpfaObj.T * ones(1, gpfaObj.L);
            gpfaObj.Cov = k - k * G * blockmldivide((I + k * G), blocks, k);
            try
                A = chol(gpfaObj.Cov);
            catch
                keyboard
            end
            % gpfaObj.Cov = k - k * G * ((I + k * G) \ k);
        end
        
        function gpfaObj = updateKernelF(gpfaObj)
            % Note: Kf is only [S x S]. The prior covariance per neuron is Kf*signs(n)^2.
            % Adding a small diagonal component for stability
            gpfaObj.Kf = exp(-gpfaObj.ss2 / gpfaObj.tauf^2) + (1e-6)*eye(size(gpfaObj.ss2));
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
    end
end