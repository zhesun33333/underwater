function write_bell( fid, Beam, model )

% Write out the rest of the environmental file
% This is the part containing control info specific to Bellhop

% *** write run type ***
fprintf( fid, '''%s'' \t \t \t \t ! Run Type \n', Beam.RunType );

%%
% declination angles
if ( isfield( Beam, 'Ibeam' ) )
   fprintf( fid, '%5i %i \t \t \t \t ! Nbeams Ibeam \n', Beam.Nbeams, Beam.Ibeam );
else
   % if this is a ray trace run and the field Beam.Nrays exists to use
   % fewer rays in the trace, then use that
   if ( Beam.RunType( 1 : 1 ) == 'R' && isfield( Beam, 'Nrays' ) )
      fprintf( fid, '%5i \t \t \t \t \t ! Nbeams \n', Beam.Nrays );
   else
      fprintf( fid, '%5i \t \t \t \t \t ! Nbeams \n', Beam.Nbeams );
   end
end

fprintf( fid, '%f %f / \t \t \t \t ! declination angles (degrees) \n', Beam.alpha( 1 ), Beam.alpha( end ) );

%%
if ( strcmp( model, 'BELLHOP3D' ) )
   % azimuthal angles
   fprintf( fid, '%5i \t \t \t \t \t ! Nbeta \n', Beam.Nbeta );
   fprintf( fid, '%f %f / \t \t \t \t ! azimuthal angles (degrees) \n', Beam.beta( 1 ), Beam.beta( end ) );
end

%%

if ( strcmp( model, 'BELLHOP' ) )
   fprintf( fid, '%f %f %f    \t ! deltas (m) Box.z (m) Box.r (km) \n', Beam.deltas, Beam.Box.z, Beam.Box.r );
else
   fprintf( fid, '%f %f %f %f \t ! deltas (m) Box.x (km) Box.y (km) Box%z (m) \n', Beam.deltas, Beam.Box.x, Beam.Box.y, Beam.Box.z );
end

% Cerveny-style Gaussian beams
if ( length( Beam.RunType ) > 1 &&  ~contains( 'GBS', Beam.RunType(2:2) )  )
   fprintf( fid, '''%s'' %f %f  \t \t \t ! ''Min/Fill/Cer, Sin/Doub/Zero'' Epsmult RLoop (km) \n', Beam.Type(1:2), Beam.epmult, Beam.rLoop );
   fprintf( fid, '%i %i  \t \t \t \t ! Nimage Ibwin \n', Beam.Nimage, Beam.Ibwin );
end