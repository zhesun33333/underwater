function cleanenv( infile, outfile, model )

% reads the standard envfil and outputs it in a clean format with
% standardized commenting
%
% usage: cleanenv( infile, outfile, model )
% where filename is the environmental file (without the extension)
% Filename should be enclosed in quotes if given as a literal

[ TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax, fid ] = read_env( infile, model );

write_env( outfile, model, TitleEnv, freq, SSP, Bdry, Pos, Beam, cInt, RMax )
