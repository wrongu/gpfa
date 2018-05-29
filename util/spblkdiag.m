function M = spblkdiag(varargin)
%SPBLKDIAG like the built-in blkdiag, but guarantees that result is sparse.

nBlocks = length(varargin);
rowHeights = cellfun(@(mat) size(mat, 1), varargin);
colWidths = cellfun(@(mat) size(mat, 2), varargin);

Mcell = cell(nBlocks, nBlocks);
for i=1:nBlocks
    for j=1:nBlocks
        if i == j
            Mcell{i,j} = varargin{i};
        else
            Mcell{i,j} = sparse(rowHeights(i), colWidths(j));
        end
    end
end

% Note that cell2mat returns a sparse result if any of the constituent matrices is sparse.
M = cell2mat(Mcell);

end