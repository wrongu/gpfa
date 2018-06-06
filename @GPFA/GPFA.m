classdef GPFA
    %GPFA class wrapping the parameters and state of a GPFA model.
    
    properties
        %% --- Matrices ---
        Y % [T x N] data points
        C % [N x L] latents loadings
        R % [N x 1] private variance of each neuron (full covariance is diag(R))
        S % [T x M] stimulus values (optional - may be empty)
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
    end
    
    properties% (Access = protected)
        %% --- Metadata ---
        isKernelToeplitz % whether K matrix has toeplitz structure (requires equal-spaced time points)
%         preTransform     % preprocessing applied to data (e.g. @sqrt to square-root transform spike counts)
%         postTransform    % the inverse of preTransform
        %% --- Precomputed matrices ---
        K     % [TL x TL] (sparse matrix) kernel-based covariance for (flattened) latents
        Gamma % [TL x TL] (sparse matrix) Kronecker of (C'*inv(R)*C) and eye(T), adjusted for missing data.
        Cov   % Posterior covariance matrix, inv(inv(K) + Gamma), computed stably for the case when inv(K) is poorly conditioned
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
            %% Copy fields from varargin
            allProps = properties(gpfaObj);
            for i=1:2:length(varargin)
                fieldname = varargin{i};
                % Note: isfield() does not work on objects, but searching for property names does.
                if any(strcmp(allProps, fieldname))
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
            assert(isempty(gpfaObj.times) || length(gpfaObj.times) == gpfaObj.T, '''times'' must be size [N x 1]');
            assert(isempty(gpfaObj.dt) || isscalar(gpfaObj.dt), '''dt'' must be a scalar');
            
            if ~isempty(gpfaObj.S)
                if isempty(gpfaObj.M), gpfaObj.M = size(gpfaObj.S, 2); end
                assert(size(gpfaObj.S, 1) == gpfaObj.T, '''S'' must be size [T x M]');
                assert(isempty(gpfaObj.D) || all(size(gpfaObj.D) == [gpfaObj.N gpfaObj.M]), '''D'' must be size [N x M]');
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
            end
            
            if isempty(gpfaObj.taus)
                gpfaObj.taus = 10 * effectiveDt * ones(1, gpfaObj.L);
            end
            
            if isempty(gpfaObj.sigs)
                gpfaObj.sigs = ones(1, gpfaObj.L);
            end
            
            if isempty(gpfaObj.rhos)
                % Note: small nonzero rho helps numerical stability
                gpfaObj.rhos = zeros(1, gpfaObj.L);
            end
            
            %% Check for and apply preprocessing transformations
            
%             if ~isempty(gpfaObj.preTransform)
%                 % Transform Y data
%                 gpfaObj.Y = gpfaObj.preTransform(gpfaObj.Y);
%                 
%                 % If not given, try to automatically infer what 'postTransform' should be.
%                 if isempty(gpfaObj.postTransform)
%                     if isequal(gpfaObj.preTransform, @sqrt)
%                         gpfaObj.postTransform = @(x) x.^2;
%                     elseif isequal(gpfaObj.preTransform, @log)
%                         gpfaObj.postTransform = @exp;
%                     else
%                         error('Not sure how to invert ''%s''. Supply your own ''postTransform''', func2str(gpfaObj.preTransform));
%                     end
%                 end
%             elseif ~isempty(gpfaObj.postTransform)
%                 error('''postTransform'' is given without any ''preTransform''');
%             end
            
            
            %% Initialize loadings if they were not provided
            gpfaInit = gpfaObj.initialize();
            
            if isempty(gpfaObj.b), gpfaObj.b = gpfaInit.b; end
            if isempty(gpfaObj.C), gpfaObj.C = gpfaInit.C; end
            if isempty(gpfaObj.D), gpfaObj.D = gpfaInit.D; end
            if isempty(gpfaObj.R), gpfaObj.R = gpfaInit.R; end
            
            % Initialize all 'precomputed' matrices
            gpfaObj = gpfaObj.updateAll();
        end
        
        %% Inference
        [mu_x, sigma_x] = inferX(gpfaObj, Y)
        
        %% Learning
        [gpfaObj, Q] = emStep(gpfaObj, fixedParams)
        [bestFit, Qs] = fitEM(gpfaObj, maxIters, convergenceTol, fixedParams)

        %% Simulation / Generate Data
        [Yhat, x] = simulate(gpfaObj)
    end
    
    methods (Access = protected)
        %% Helper to initialize parameters based on data
        function gpfaObj = initialize(gpfaObj)
            % Initialize mean b using mean of data
            gpfaObj.b = nanmean(gpfaObj.Y, 1)';
            
            % Initialize latent loadings C using top L principal components
            dataCov = nancov(gpfaObj.Y, 'pairwise');
            [gpfaObj.C, ~] = eigs(dataCov, gpfaObj.L);
            
            % Initialize stimulus loadings D using linear regression
            if ~isempty(gpfaObj.S)
                % The following is the same as first replacing each missing value with the mean
                % (since b is the mean) then regressing using (Y-b)/S
                Yeffective = gpfaObj.Y' - gpfaObj.b;
                Yeffective(isnan(Yeffective)) = 0;
                % mrdivide with [N x T] / [M x T] results in a [N x M] matrix D
                gpfaObj.D = Yeffective / gpfaObj.S';
            end
            
            % Initialize private variance R using residuals from the stimulus prediction only,
            % scaled up by 10 because over-estimating variance early helps keep EM stable.
            residuals = gpfaObj.Y - gpfaObj.b';
            if ~isempty(gpfaObj.S)
                residuals = residuals - gpfaObj.S * gpfaObj.D';
            end
            gpfaObj.R = 10 * nanvar(residuals, [], 1)';
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
%             blocks = gpfaObj.T * ones(1, gpfaObj.L);
%             gpfaObj.Cov = k - k * G * blockmldivide((I + k * G), blocks, k); 
            gpfaObj.Cov = k - k * G * ((I + k * G) \ k); 
        end
        
        function gpfaObj = updateAll(gpfaObj, Y)
            if exist('Y', 'var')
                gpfaObj = updateCov(updateK(updateGamma(gpfaObj, Y)));
            else
                gpfaObj = updateCov(updateK(updateGamma(gpfaObj)));
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
            end
        end
    end
end